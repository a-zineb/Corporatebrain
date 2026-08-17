import time
import streamlit as st
import os
import chromadb
import re
import hashlib
import json
import io
import zipfile
import torch
from datetime import datetime
import pandas as pd
import docx
import fitz  # PyMuPDF
import docx2txt
import olefile
import struct
import unicodedata

import rag_pipeline
import canonical_rag
import mvp_services
import ui_components
from backend.llm import ProviderError, get_generation_provider
from structured_ingestion import build_structured_docx_index_payload

generation_provider = get_generation_provider()


def _env_flag(name: str) -> bool:
    """Return True only for the literal, case-insensitive value ``true``."""
    return os.getenv(name, "").strip().casefold() == "true"


ENABLE_STRUCTURED_DOCX_INGESTION = _env_flag("ENABLE_STRUCTURED_DOCX_INGESTION")
STRUCTURED_INGESTION_DRY_RUN = _env_flag("STRUCTURED_INGESTION_DRY_RUN")
ENABLE_GENERIC_STRUCTURED_DIRECT = _env_flag("ENABLE_GENERIC_STRUCTURED_DIRECT")
ENABLE_STARTUP_SYNC = _env_flag("ENABLE_STARTUP_SYNC")
ENABLE_CANONICAL_DEBUG = _env_flag("ENABLE_CANONICAL_DEBUG")
DIRECT_DEBUG = _env_flag("DIRECT_DEBUG")
DIRECT_FACT_DEBUG = _env_flag("DIRECT_FACT_DEBUG")

CANONICAL_NO_EVIDENCE_FR = (
    "Je n’ai trouvé aucune preuve explicite répondant à cette question "
    "dans le document sélectionné."
)


def canonical_no_evidence_message(language):
    if language == "English":
        return "I found no explicit evidence answering this question in the selected document."
    if language == "Spanish":
        return "No encontré pruebas explícitas que respondan a esta pregunta en el documento seleccionado."
    return CANONICAL_NO_EVIDENCE_FR


# Fonction pour ouvrir un fichier local
def open_local_file(path):
    try:
        if os.path.exists(path):
            os.startfile(path)
    except Exception as e:
        print(f"Erreur d'ouverture de fichier : {e}")


def extractive_answers_enabled():
    """Return whether the opt-in production extractive route is enabled."""
    return os.getenv("EXTRACTIVE_FACTUAL_ANSWERS_ENABLED", "").strip().casefold() == "true"


def detect_direct_factual_intent(query, has_history=False):
    """Conservatively route only standalone, single-fact questions."""
    if has_history:
        return False
    text = " ".join(str(query or "").split()).strip()
    if not text or len(text) > 240:
        return False
    lowered = text.casefold()
    excluded = (
        "summary", "summarize", "résumé", "resume", "explain", "expliquer",
        "compare", "compar", "pourquoi", "why", "how", "comment",
        "recommend", "recommand", "list all", "tous les documents",
        "catalogue", "catalog", "et ", " and ", " ou ", " or ",
    )
    if any(marker in lowered for marker in excluded):
        return False
    if re.search(r"\b(it|this|that|these|those|ceci|cela|ca|cette|ce)\b", lowered):
        return False
    if "?" not in text and not re.match(
        r"^(what|which|where|who|when|how many|quel|quelle|quels|quelles|où|qui|quand|combien)",
        lowered,
    ):
        return False
    factual_prefix = re.match(
        r"^(what|which|where|who|when|how many|quel|quelle|quels|quelles|où|qui|quand|combien)",
        lowered,
    )
    return bool(factual_prefix) and len(re.findall(r"(?:and|et|or|ou)", lowered)) == 0

# Replace the declarative helper above with a clean ASCII-safe regex version.
def detect_direct_factual_intent(query, has_history=False):
    if has_history:
        return False
    text = " ".join(str(query or "").split()).strip()
    if not text or len(text) > 240:
        return False
    lowered = text.casefold()
    excluded = (
        "summary", "summarize", "resume", "explain", "expliquer",
        "compare", "compar", "pourquoi", "why", "how", "comment",
        "recommend", "recommand", "list all", "tous les documents",
        "catalogue", "catalog", "et ", " and ", " ou ", " or ",
    )
    if any(marker in lowered for marker in excluded):
        return False
    if re.search(r"\b(it|this|that|these|those|ceci|cela|ca|cette|ce)\b", lowered):
        return False
    prefix = r"^(what|which|where|who|when|how many|quel|quelle|quels|quelles|où|ou|qui|quand|combien)\b"
    return bool(re.match(prefix, lowered)) and len(re.findall(r"\b(?:and|et|or|ou)\b", lowered)) == 0


def is_direct_answer_suitable(query):
    """Return whether Direct answer can safely answer this query extractively."""
    text = " ".join(str(query or "").split()).strip()
    if not text or len(text) > 240:
        return False
    normalized = normalize_catalog_text(text)
    unsuitable = (
        "explain", "expliquer", "explique", "pourquoi", "por que", "porque", "why", "compare", "comparaison",
        "difference", "differences", "resume", "summarize", "overview", "synthese",
        "analyze", "analyse", "interpret", "interprete", "recommend", "recommand",
        "architecture", "broad", "comment fonctionne", "whole workflow",
    )
    if any(marker in normalized for marker in unsuitable):
        return False
    # In structured specific-document mode, interrogative how/is/does forms
    # are factual candidates; the extractor, not this gate, decides evidence.
    if re.match(r"^(how does|how do|how are|is|are|does|can)\b", normalized):
        return True
    technical_attribute_markers = (
        "version", "filename pattern", "duplicate detection", "duplicate check", "duplicate files", "duplicates", "parameter",
        "configuration parameter", "input directory", "output directory", "folder", "path",
        "cache age", "duration", "schedule", "frequency", "table", "server", "hostname",
        "protocol", "port", "modele de fichier", "detection des doublons", "parametre",
        "how often", "a quelle frequence", "tous les combien",
        "repertoire d entree", "repertoire de sortie", "dossier", "chemin", "duree",
        "horario", "frecuencia", "servidor", "protocolo", "puerto", "patron de nombre de archivo", "patron",
    )
    technical_attribute_query = any(marker in normalized for marker in technical_attribute_markers)
    factual_how = re.search(
        r"\bhow\b.*\b(?:detected|detecte|often|frequently|protocol|collected|files|transformed|many)\b",
        normalized,
    )
    if re.search(r"\bhow\b", normalized) and "how many" not in normalized and not technical_attribute_query and not factual_how:
        return False
    concise_factual_patterns = (
        r"\b(?:total|number of|count|nombre total|numero de|combien)\b",
        r"\b(?:what floor|which floor|located|where|ou|a quel etage|que piso)\b",
        r"\b(?:opening hours|hours|when open|horaires|heure d ouverture|horario|horarios)\b",
        r"\b(?:what approval|who approves|quelle approbation|que aprobacion|aprobacion)\b",
        r"\b(?:maximum(?:\s+\w+){0,2}\s+duration|cache age|duree maximale|duracion maxima)\b",
        r"\b(?:version|specification version|version de especificacion)\b",
        r"\b(?:filename pattern|duplicate detection|duplicate check|duplicate files|parameter|configuration parameter|input directory|output directory|folder|path|cache age|duration|schedule|frequency|table|server|hostname|protocol|port)\b",
        r"\b(?:modele de fichier|detection des doublons|parametre|repertoire d entree|repertoire de sortie|dossier|chemin|duree|frequence|a quelle frequence|tous les combien|how often|serveur|protocole|puerto|patron de nombre de archivo|patron)\b",
        r"\b(?:input workflows?|workflows?|collection server|server ip|output transformed|transformation|post[- ]collection|audit tables?|archive modes?)\b",
    )
    if any(re.search(pattern, normalized) for pattern in concise_factual_patterns):
        return True
    # Inverted factual French forms remain suitable for deterministic lookup.
    if re.search(r"\b(?:comporte|contient|compte)\b.*\bcombien\b", normalized):
        return True
    if re.match(r"^a quel etage\b", normalized):
        return True
    factual_prefix = re.match(
        r"^(who|where|when|which|what|how many|qui|ou|quand|quel|quelle|quels|quelles|combien)\b",
        normalized,
    )
    if not factual_prefix:
        return False
    conjunctions = re.findall(r"\b(?:and|et)\b", normalized)
    if conjunctions:
        interrogatives = re.findall(
            r"\b(?:who|where|when|which|what|how many|qui|ou|quand|quel|quelle|quels|quelles|combien)\b",
            normalized,
        )
        target_pair = (
            ("location" in normalized or "situe" in normalized or "trouve" in normalized)
            and ("hour" in normalized or "horaire" in normalized or "ouverture" in normalized)
        )
        if len(interrogatives) < 2 and not target_pair:
            return False
    return True


def is_obvious_synthesis_query(query):
    """Detect explicit explanation/synthesis requests before structured routing."""
    normalized = normalize_catalog_text(query)
    markers = (
        "why", "pourquoi", "por que", "porque", "explain", "expliquer", "explique",
        "summarize", "summary", "resume", "résume", "compare", "comparaison",
        "analyze", "analyse", "interpret", "interprete", "recommend", "recommand",
        "overview", "whole workflow", "entire workflow", "whole architecture", "entire architecture",
    )
    return any(marker in normalized for marker in markers)


def direct_unsuitable_message(language):
    if str(language or "").casefold() == "english":
        return "This question requires explanation or synthesis. Use \"AI answer\" mode for a generated response based on the documents."
    if str(language or "").casefold() == "spanish":
        return "Esta pregunta requiere una explicación o una síntesis. Utiliza el modo «AI answer» para obtener una respuesta generada a partir de los documentos."
    return "Cette question nécessite une explication ou une synthèse. Utilisez le mode « AI answer » pour obtenir une réponse générée à partir des documents."


def detect_direct_vague_entity(query):
    """Return the named entity for an entity-only factual query."""
    normalized = normalize_catalog_text(query)
    if len(normalized.split()) != 1:
        return None
    entities = (
        ("INZsmart", "inzsmart"), ("SIMBOX", "simbox"), ("VPN", "vpn"),
        ("MBF", "mbf"), ("cafeteria", "cafeteria"), ("GGSN", "ggsn"),
        ("P2P", "p2p"), ("CRBT", "crbt"), ("Huawei MSC", "huawei msc"),
    )
    for display, token in entities:
        if normalized == token:
            match = re.search(re.escape(token), str(query or ""), flags=re.IGNORECASE)
            return match.group(0) if match else display
    return None


def direct_vague_message(language, entity):
    entity = entity or ""
    if str(language or "").casefold() == "english":
        return f"What would you like to know about {entity}?"
    if str(language or "").casefold() == "spanish":
        return f"Â¿Qué desea saber sobre {entity}?"
    return f"Que souhaitez-vous savoir sur {entity} ?"


def detect_direct_incomplete_query(query):
    """Recognize follow-up fragments that lack a factual subject."""
    normalized = normalize_catalog_text(query)
    incomplete = {
        "definition", "meaning", "the answer", "what about it", "and the version",
        "et la duree", "et la durée", "y la version",
    }
    return normalized.rstrip(" ?!.:") in {normalize_catalog_text(value) for value in incomplete}


def direct_incomplete_message(language):
    if str(language or "").casefold() == "english":
        return "Please restate the complete factual question and include the subject."
    if str(language or "").casefold() == "spanish":
        return "Reformule la pregunta factual completa e indique el tema."
    return "Veuillez reformuler la question factuelle complète en précisant le sujet."


def is_direct_sensitive_request(query):
    """Detect direct requests for credentials or other authentication secrets."""
    normalized = normalize_catalog_text(query)
    markers = (
        "password", "passwd", "mot de passe", "credential", "credentials", "identifiant",
        "api key", "access key", "secret key", "token", "bearer", "jwt", "private key",
        "cle privee", "clé privée", "secret", "secrets", "authentication value",
        "authentication values", "valeur d authentification", "valeurs d authentification",
        "social security number", "social security numbers", "numero de securite sociale",
        "national identification number", "national identification numbers",
        "numero national d identification", "identifiant national",
        "identity card number", "identity card numbers", "numero de carte d identite",
        "personal identification number", "personal identification numbers",
        "numero d identification personnelle", "numero de identificacion personal",
        "numero de seguridad social", "numero nacional de identificacion",
        "numero de documento de identidad", "cin",
    )
    has_secret = any(marker in normalized for marker in markers)
    if not has_secret:
        return False
    policy_markers = (
        "policy", "policies", "requirements", "requirement", "rules", "rule",
        "minimum length", "complexity", "expiration", "rotation", "renewal", "changed", "change",
        "mfa", "multi factor", "authentication procedure", "politique",
        "exigences", "regles", "longueur minimale", "complexite", "expiration",
        "rotation", "renouvellement", "authentification multifacteur",
        "politica", "requisitos", "reglas", "longitud minima", "renovacion",
    )
    if any(marker in normalized for marker in policy_markers):
        secret_verbs = (
            "show", "reveal", "provide", "give", "display", "retrieve", "tell",
            "donne moi", "affiche", "montre", "revela", "proporciona", "dame",
        )
        if not any(marker in normalized for marker in secret_verbs):
            return False
    return True


def contains_sensitive_output(text):
    """Detect secret-like values without returning or logging the value."""
    value = str(text or "")
    password_pattern = re.compile(
        r"(?i)\b(?:password|passwd|mot de passe)\s*[:=]\s*(?P<secret>\S+)"
    )
    for match in password_pattern.finditer(value):
        if match.group("secret").casefold() not in {"[redacted]", "***", "******"}:
            return True
    patterns = (
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|token|bearer|jwt|secret)\s*[:=]\s*\S+",
        r"(?i)\b(?:username|user)\s*[:=]\s*\S+\s+(?:password|passwd)\s*[:=]\s*\S+",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    return any(re.search(pattern, value) for pattern in patterns)


def direct_sensitive_message(language):
    if str(language or "").casefold() == "english":
        return "I can’t provide passwords, credentials, keys, tokens, or other authentication secrets."
    if str(language or "").casefold() == "spanish":
        return "No puedo proporcionar contraseñas, credenciales, claves, tokens u otros secretos de autenticación."
    return "Je ne peux pas fournir de mots de passe, identifiants, clés, jetons ou autres secrets d’authentification."


def detect_catalog_intent(query):
    """Recognize explicit requests for the indexed knowledge catalog."""
    lowered = normalize_catalog_text(query)
    markers = (
        "all resources", "all the resources", "all documents", "indexed documents", "knowledge catalog",
        "catalogue de connaissances", "catalogue", "list all", "show the catalog",
        "tous les documents", "toutes les ressources", "ressources disponibles",
        "what documents do you have", "what files do you have", "documents do you have",
    )
    return any(marker in lowered for marker in markers)


def normalize_catalog_text(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    text = "".join(
        character for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def parse_catalog_refinements(query):
    """Parse deterministic file-type, metadata, and filename/topic refinements."""
    normalized = normalize_catalog_text(query)
    if re.search(r"(all|tous|toutes|all of them|tout)", normalized):
        return {"clear": True, "file_types": [], "terms": [], "metadata": {}}
    file_types = []
    if re.search(r"(pdf|pdfs)", normalized):
        file_types.append("pdf")
    if re.search(r"(doc|docx|word|words)", normalized):
        file_types.extend(["doc", "docx"])
    if re.search(r"(xls|xlsx|excel|excels)", normalized):
        file_types.extend(["xls", "xlsx"])
    metadata = {}
    for zone in ("ocm", "oeg", "ojo", "oci"):
        if re.search(rf"{zone}", normalized):
            metadata["geographical_entity"] = zone.upper()
    for application in ("kpsa", "mz"):
        if re.search(rf"{application}", normalized):
            metadata["application"] = application.upper()
    stop = {
        "only", "the", "a", "an", "all", "of", "them", "files", "file", "documents",
        "document", "docs", "resources", "resource", "show", "give", "me", "that", "are", "in", "here", "do", "you",
        "have", "what", "list", "indexed", "knowledge", "catalog", "catalogue",
        "pdf", "pdfs", "doc", "docx", "word", "words", "xls", "xlsx", "excel", "excels",
        "ocm", "oeg", "ojo", "oci", "kpsa", "mz",
    }
    terms = [term for term in normalized.split() if term not in stop and len(term) > 1]
    return {
        "clear": False,
        "file_types": sorted(set(file_types)),
        "terms": sorted(set(terms)),
        "metadata": metadata,
    }


def merge_catalog_refinements(previous, current, continuation=False):
    """Reuse prior catalog state only for explicit continuations."""
    if not continuation:
        return {
            "file_types": list(current.get("file_types", [])),
            "terms": list(current.get("terms", [])),
            "metadata": dict(current.get("metadata", {})),
        }
    if current.get("clear"):
        return {}
    return {
        "file_types": list(current.get("file_types") or previous.get("file_types", [])),
        "terms": list(current.get("terms") or previous.get("terms", [])),
        "metadata": {
            **previous.get("metadata", {}),
            **current.get("metadata", {}),
        },
    }

def parse_catalog_refinements(query):
    """Parse catalog refinements using normalized token matching."""
    normalized = normalize_catalog_text(query)
    tokens = set(normalized.split())
    if {"all", "of", "them"}.issubset(tokens) or tokens.intersection({"tous", "toutes", "tout"}):
        return {"clear": True, "file_types": [], "terms": [], "metadata": {}}
    file_types = []
    if tokens.intersection({"pdf", "pdfs"}):
        file_types.append("pdf")
    if tokens.intersection({"doc", "docx", "word", "words"}):
        file_types.extend(["doc", "docx"])
    if tokens.intersection({"xls", "xlsx", "excel", "excels"}):
        file_types.extend(["xls", "xlsx"])
    metadata = {}
    for zone in ("ocm", "oeg", "ojo", "oci"):
        if zone in tokens:
            metadata["geographical_entity"] = zone.upper()
    for application in ("kpsa", "mz"):
        if application in tokens:
            metadata["application"] = application.upper()
    stop = {
        "only", "the", "a", "an", "all", "of", "them", "files", "file", "documents",
        "document", "docs", "resources", "resource", "show", "give", "me", "that", "are", "in", "here", "do", "you",
        "have", "what", "list", "indexed", "knowledge", "catalog", "catalogue",
        "pdf", "pdfs", "doc", "docx", "word", "words", "xls", "xlsx", "excel", "excels",
        "ocm", "oeg", "ojo", "oci", "kpsa", "mz",
    }
    terms = [term for term in tokens if term not in stop and len(term) > 1]
    return {
        "clear": False,
        "file_types": sorted(set(file_types)),
        "terms": sorted(set(terms)),
        "metadata": metadata,
    }


def detect_catalog_continuation(query, previous_actual_mode=None):
    """Keep catalog follow-ups in catalog mode after a catalog response."""
    if previous_actual_mode != "catalog":
        return False
    lowered = normalize_catalog_text(query)
    if any(marker in lowered for marker in (
        "explain", "expliquer", "why", "pourquoi", "how does", "comment fonctionne",
        "compare", "compar", "summary", "résumé", "resume",
    )):
        return False
    continuation_markers = (
        "files", "file", "documents", "document", "all of them", "show them",
        "those documents", "only those pdfs", "only those pdf", "only the pdfs",
        "only the pdf documents", "only pdf", "give me the files",
        "the files that are in here", "fichiers", "documents", "tous", "toutes",
        "ceux-là", "ceux la", "montre-les", "uniquement les pdf",
    )
    return any(marker in lowered for marker in continuation_markers)


# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Corporate Brain", layout="wide")
ui_components.inject_design()

# ==========================================
# 2. CONFIGURATION INITIALE & DOSSIERS
# ==========================================
STORAGE_DIR = "doc_storage_v2"
CHROMA_PATH = "chroma_db_local_v2"
COLLECTION_NAME = "documents"

# Garantir que les dossiers existent avant toute initialisation
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# ==========================================
# 3. CHARGEMENT DU BACKEND (EMBEDDING)
# ==========================================
@st.cache_resource
def load_backend():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    embedding_model = rag_pipeline.load_embedding_model_offline(
        rag_pipeline.RAGConfig(
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
        )
    )

    return client, collection, embedding_model

try:
    with st.spinner(" Chargement du moteur de recherche ultra-rapide (Local)..."):
        client, collection, embedding_model = load_backend()
    # Test rapide pour vérifier que la collection est valide
    collection.count()
except Exception:
    # La collection a été réinitialisée manuellement : on vide le cache et on recharge
    load_backend.clear()
    with st.spinner(" Réinitialisation de la base de données..."):
        client, collection, embedding_model = load_backend()

@st.cache_resource
def build_bm25_index(_collection, _count):
    return rag_pipeline.build_bm25_index(_collection, _count)


def hybrid_search(query, collection, embedding_model, bm25, docs, metadatas, chroma_filter=None, top_k=5,
                   min_results_before_relax=3):
    return rag_pipeline.hybrid_search(
        query=query,
        collection=collection,
        embedding_model=embedding_model,
        bm25=bm25,
        docs=docs,
        metadatas=metadatas,
        chroma_filter=chroma_filter,
        top_k=top_k,
        min_results_before_relax=min_results_before_relax,
    ).as_legacy_tuple()


def list_catalog_documents(collection, chroma_filter=None, refinements=None):
    """Read every in-scope document metadata record without retrieval."""
    records = collection.get(where=chroma_filter, include=["metadatas"]).get("metadatas", [])
    unique = {}
    for metadata in records:
        if not isinstance(metadata, dict):
            continue
        filename = metadata.get("source_file")
        if not isinstance(filename, str) or not filename:
            continue
        unique.setdefault(metadata.get("file_hash") or filename, metadata)
    rows = list(unique.values())
    refinements = refinements or {}
    file_types = set(refinements.get("file_types", ()))
    terms = tuple(refinements.get("terms", ()))
    metadata_filters = refinements.get("metadata", {})
    if metadata_filters:
        rows = [
            row for row in rows
            if all(
                str(row.get(key, "")).casefold() == str(value).casefold()
                for key, value in metadata_filters.items()
            )
        ]
    if file_types:
        rows = [
            row for row in rows
            if os.path.splitext(str(row.get("source_file", "")))[1]
            .lstrip(".").casefold() in file_types
        ]
    if terms:
        rows = [
            row for row in rows
            if all(
                term in normalize_catalog_text(" ".join(str(value) for value in row.values()))
                for term in terms
            )
        ]
    return sorted(rows, key=lambda item: (
        item.get("application", ""),
        item.get("geographical_entity", ""),
        item.get("source_file", "").lower(),
    ))


def direct_document_identity(metadata):
    """Return the stable document identity, preferring file_hash."""
    return str(metadata.get("file_hash") or metadata.get("source_file") or "")


def build_direct_document_filter(active_filter, metadata):
    """Combine active sidebar scope with the selected document identity."""
    key = "file_hash" if metadata.get("file_hash") else "source_file"
    condition = {key: metadata.get(key)}
    if not active_filter:
        return condition
    if "$and" in active_filter:
        return {"$and": [*active_filter["$and"], condition]}
    return {"$and": [active_filter, condition]}


def direct_filter_contains_identity(chroma_filter, metadata):
    """Return whether an effective filter pins retrieval to this document."""
    key = "file_hash" if metadata.get("file_hash") else "source_file"
    expected = metadata.get(key)
    if not expected or not isinstance(chroma_filter, dict):
        return False
    conditions = chroma_filter.get("$and", [chroma_filter])
    return any(
        isinstance(condition, dict) and condition.get(key) == expected
        for condition in conditions
    )


def direct_metadata_matches_identity(metadata, selected_metadata):
    """Enforce the selected document identity on every returned chunk/source."""
    if not isinstance(metadata, dict) or not isinstance(selected_metadata, dict):
        return False
    selected_hash = selected_metadata.get("file_hash")
    if selected_hash:
        return metadata.get("file_hash") == selected_hash
    selected_file = selected_metadata.get("source_file")
    return bool(selected_file) and metadata.get("source_file") == selected_file


def direct_scope_audit_filter(chroma_filter):
    """Serialize only sanitized metadata-filter structure for audit records."""
    return json.dumps(chroma_filter, sort_keys=True, ensure_ascii=False)


def direct_invalid_scope_message(language):
    return {
        "English": "Please reselect the document before using Direct answer.",
        "Spanish": "Vuelve a seleccionar el documento antes de usar Direct answer.",
    }.get(language, "Veuillez resélectionner le document avant d'utiliser Direct answer.")


def direct_scope_selection_consistent(selected_scope, session_scope):
    """Ensure the rendered selector and effective execution scope agree."""
    return selected_scope == session_scope


def experimental_global_direct_answer_enabled():
    """Expose global Direct Answer scope only when explicitly enabled."""
    return os.getenv("ENABLE_EXPERIMENTAL_GLOBAL_DIRECT_ANSWER", "").strip().casefold() == "true"


def resolve_direct_document(collection, active_filter, document_id):
    """Resolve and validate a selected document inside the active filter scope."""
    if not document_id:
        return None
    for metadata in list_catalog_documents(collection, active_filter):
        if direct_document_identity(metadata) == document_id:
            return metadata
    return None


def structured_specific_direct_answer_enabled():
    """Enable exhaustive structured selection only for the approved opt-in path."""
    return os.getenv("ENABLE_STRUCTURED_DOCX_INGESTION", "").strip().casefold() == "true"


def generic_structured_direct_answer_enabled():
    """Enable schema-driven structured evidence only behind an explicit flag."""
    return os.getenv("ENABLE_GENERIC_STRUCTURED_DIRECT", "").strip().casefold() == "true"


def fetch_structured_specific_chunks(collection, selected_document):
    """Fetch all chunks for one selected document, rejecting mixed identities."""
    if not isinstance(selected_document, dict):
        return None
    key = "file_hash" if selected_document.get("file_hash") else "source_file"
    value = selected_document.get(key)
    if not value:
        return None
    data = collection.get(where={key: value}, include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if len(documents) != len(metadatas):
        return None
    if not metadatas or not all(direct_metadata_matches_identity(meta, selected_document) for meta in metadatas):
        return None
    if not any(meta.get("block_type") and meta.get("chunk_ordinal") is not None for meta in metadatas):
        return None
    return [
        rag_pipeline.ChunkRecord(text=text, metadata=meta)
        for text, meta in zip(documents, metadatas)
    ]


def selected_document_has_structured_metadata(collection, selected_document):
    """Distinguish legacy chunks from structured chunks before routing."""
    if not isinstance(selected_document, dict):
        return False
    key = "file_hash" if selected_document.get("file_hash") else "source_file"
    value = selected_document.get(key)
    if not value:
        return False
    data = collection.get(where={key: value}, include=["metadatas"])
    metadatas = data.get("metadatas") or []
    return bool(metadatas) and any(
        meta.get("block_type") and meta.get("chunk_ordinal") is not None
        for meta in metadatas
    )


# ==========================================
# 4. FONCTIONS UTILITAIRES (DÉTECTION, MODELS, REFORMULATION)
# ==========================================

def detect_query_language(text, fallback_lang="French"):
    import unicodedata as _unicodedata

    normalized = _unicodedata.normalize("NFKD", str(text or "")).casefold()
    text_lower = "".join(character for character in normalized if not _unicodedata.combining(character)).strip()

    en_keywords = {'what', 'how', 'why', 'who', 'where', 'when', 'is', 'are', 'the', 'this', 'that', 'can', 'you', 'explain', 'tell', 'me', 'show', 'list', 'details', 'about', 'english', 'mean', 'stand', 'eenglish'}
    fr_keywords = {'que', 'quoi', 'comment', 'pourquoi', 'qui', 'ou', 'quand', 'est', 'sont', 'le', 'la', 'les', 'ce', 'cette', 'peux', 'tu', 'expliquer', 'montre', 'donne', 'veut', 'dire', 'quel', 'quelle', 'quels', 'quelles', 'combien', 'horaires', 'duree', 'age', 'instances'}

    words = set(re.findall(r'\b\w+\b', text_lower))
    en_score = len(words.intersection(en_keywords))
    fr_score = len(words.intersection(fr_keywords))
    es_keywords = {'que', 'como', 'por', 'quien', 'donde', 'cuando', 'cuantas', 'cuantos', 'esta', 'estan', 'el', 'la', 'los', 'las', 'cuanto', 'dame', 'muestra'}
    es_score = len(words.intersection(es_keywords))

    if es_score > en_score and es_score > fr_score:
        return "Spanish"
    elif en_score > fr_score:
        return "English"
    elif fr_score > en_score:
        return "French"
    return fallback_lang


def direct_answer_label(language):
    return {"English": "Answer", "Spanish": "Respuesta"}.get(language, "Réponse")


def direct_source_label(language):
    return {"English": "Source passage", "Spanish": "Pasaje fuente"}.get(language, "Passage source")


def direct_original_source_label(language):
    return {
        "English": "Original source passage",
        "Spanish": "Pasaje fuente original",
    }.get(language, "Passage source original")


def direct_clarification_message(language):
    return {
        "English": "Could you clarify the application, document, or context you mean?",
        "Spanish": "¿Podrías precisar la aplicación, el documento o el contexto?",
    }.get(language, "Pouvez-vous préciser l'application, le document ou le contexte concerné ?")


def direct_no_evidence_message(language, reason="no_explicit_evidence"):
    if reason == "missing_requested_attribute":
        if str(language or "").casefold() == "english":
            return "The selected document mentions the subject, but it does not provide the requested information."
        if str(language or "").casefold() == "spanish":
            return "El documento seleccionado menciona el tema, pero no proporciona la informaciÃ³n solicitada."
        return "Le document sÃ©lectionnÃ© mentionne le sujet, mais ne fournit pas l’information demandÃ©e."
    if str(language or "").casefold() == "english":
        return "I could not find explicit evidence for this question in the selected document."
    if str(language or "").casefold() == "spanish":
        return "No encontrÃ© evidencia explÃ­cita para esta pregunta en el documento seleccionado."
    return "Je n’ai trouvÃ© aucune preuve explicite rÃ©pondant Ã  cette question dans le document sÃ©lectionnÃ©."


def build_direct_localized_summary(query, evidence, language):
    """Build a conservative localized summary from explicit evidence values."""
    if evidence is None or not getattr(evidence, "passages", ()):
        return None
    text = " ".join(passage.text for passage in evidence.passages)
    normalized_query = normalize_catalog_text(query)
    normalized_text = normalize_catalog_text(text)
    language_key = str(language or "French").casefold()
    number_match = re.search(r"\b\d+(?:[.,]\d+)?\b", text)
    time_matches = re.findall(r"\b\d{1,2}(?:h|:)\d{2}\b", text, flags=re.IGNORECASE)
    floor_match = re.search(
        r"\b(\d+)(?:er|ère|eme|ème)?\s+(?:étage|etage|floor)\b|\b(?:floor|étage|etage)\s+(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    duration_match = re.search(r"\b(\d+(?:[.,]\d+)?)\s+(jours?|heures?|minutes?)\b", text, flags=re.IGNORECASE)
    version_match = re.search(
        r"(?:\bv(?:ersion)?\s*|\bversion\s+|\bfinal\s+version\s+)"
        r"([0-9]+(?:[.][0-9]+)+[a-z]?)\b",
        text,
        flags=re.IGNORECASE,
    )
    role_match = re.search(r"\b(manager|responsable|administrateur|directeur|équipe IT|département IT)\b", text, flags=re.IGNORECASE)
    entity = next(
        (candidate for candidate in ("INZsmart", "SIMBOX", "VPN", "cafétéria", "cafeteria")
         if normalize_catalog_text(candidate) in normalized_query),
        None,
    )

    is_count = any(term in normalized_query for term in ("combien", "how many", "nombre", "cuantas", "cuantos"))
    is_location = any(term in normalized_query for term in ("ou", "where", "etage", "floor", "location", "located", "emplacement"))
    is_opening = any(term in normalized_query for term in ("horaire", "horaires", "ouverture", "open", "opening", "when", "abierto", "horario"))
    is_duration = any(term in normalized_query for term in ("duree", "duration", "age maximal", "maximum age")) or (
        "maximum" in normalized_query and "age" in normalized_query
    )
    is_approval = any(term in normalized_query for term in ("approuve", "approval", "approve", "manager", "responsabilite"))
    is_version = "version" in normalized_query
    is_parameter = any(term in normalized_query for term in ("parametre", "parameter"))

    if is_count and number_match and entity:
        number = number_match.group(0)
        if language_key == "english":
            return f"There are {number} {entity} instances."
        if language_key == "spanish":
            return f"Hay {number} instancias de {entity}."
        return f"Il y a {number} instances {entity}."
    if is_opening and time_matches:
        start, end = time_matches[0], time_matches[-1]
        if language_key == "english":
            return f"It is open from {start} to {end}."
        if language_key == "spanish":
            return f"Está abierto de {start} a {end}."
        return f"C'est ouvert de {start} à {end}."
    if is_location and floor_match:
        floor = floor_match.group(1) or floor_match.group(2)
        if language_key == "english":
            ordinal = "th" if floor not in {"1", "2", "3"} else {"1": "st", "2": "nd", "3": "rd"}[floor]
            return f"The location is on the {floor}{ordinal} floor."
        if language_key == "spanish":
            return f"La ubicación está en el piso {floor}."
        return f"L'emplacement est au {floor}ème étage."
    if is_duration and duration_match:
        value, unit = duration_match.groups()
        if language_key == "english":
            unit = {"jours": "days", "jour": "day", "heures": "hours", "heure": "hour", "minutes": "minutes", "minute": "minute"}.get(unit.casefold(), unit)
            if entity == "SIMBOX":
                return f"The maximum SIMBOX cache age is {value} {unit}."
            return f"The duration is {value} {unit}."
        if language_key == "spanish":
            return f"La duración es de {value} {unit}."
        return f"La durée est de {value} {unit}."
    if is_approval and role_match:
        role = role_match.group(1)
        if language_key == "english":
            if entity == "VPN":
                return f"VPN requests must be approved by the {role}."
            return f"Approval is provided by the {role}."
        if language_key == "spanish":
            return f"La aprobación la proporciona {role}."
        if entity == "VPN":
            return f"Les demandes VPN doivent être approuvées par le {role}."
        return f"L'approbation est fournie par {role}."
    if is_version and version_match:
        version = version_match.group(1)
        if language_key == "english":
            if entity == "MBF":
                return f"The MBF version is {version}."
            return f"The version is {version}."
        if language_key == "spanish":
            return f"La versión es {version}."
        return f"La version est {version}."
    if is_parameter and any(term in normalized_text for term in ("duplicate", "doublon", "dupliqu", "batch check", "verification")):
        if language_key == "english":
            return "The duplicate-file detection parameter is Duplicate Batch Check."
        if language_key == "spanish":
            return "El parámetro de detección de duplicados es Duplicate Batch Check."
        return "Le paramètre de détection des doublons est Duplicate Batch Check."
    # Entity-specific French phrasing is kept deterministic and only emitted
    # when the evidence contains the requested value.
    if entity == "SIMBOX" and is_duration and duration_match and language_key not in {"english", "spanish"}:
        value, unit = duration_match.groups()
        return f"L'Ã¢ge maximal du cache SIMBOX est de {value} {unit}."
    if entity == "VPN" and is_approval and role_match and language_key not in {"english", "spanish"}:
        return f"Les demandes VPN doivent Ãªtre approuvÃ©es par le {role_match.group(1)}."


    return None


def contextualize_query(user_query, chat_history, model_name):
    return rag_pipeline.rewrite_query(
        user_query,
        chat_history,
        model_name,
        generation_provider,
    ).query

# ==========================================
# 5. EXTRACTION ET DÉCOUPE DES TEXTES
# ==========================================
def _extract_ole_doc(file_bytes):
    """
    Extrait le texte d'un vieux fichier .doc Word 97/2003 (format OLE binaire).
    Fonctionne aussi sur les faux .docx qui sont en réalité des .doc déguisés.
    Utilise olefile pour lire le flux WordDocument et une regex pour extraire
    les séquences de texte ANSI lisibles.
    """
    ole = olefile.OleFileIO(io.BytesIO(file_bytes))
    if not ole.exists('WordDocument'):
        return ''

    word_stream = ole.openstream('WordDocument').read()

    pattern = re.compile(rb'[\x20-\x7E\xC0-\xFF]{4,}')
    matches = pattern.findall(word_stream)

    text_parts = []
    for m in matches:
        try:
            text_parts.append(m.decode('latin-1'))
        except Exception:
            pass

    return '\n'.join(text_parts)

def extract_text_from_bytes(file_bytes, file_name):
    ext = os.path.splitext(file_name)[1].lower()
    text_data = []

    try:
        if ext == ".pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                text_data.append((f"Page {page_num + 1}", page.get_text()))

        elif ext == ".docx":
            try:
                doc = docx.Document(io.BytesIO(file_bytes))
                full_text = [para.text for para in doc.paragraphs]
                text_data.append(("Corps du document", "\n".join(full_text)))
            except Exception:
                try:
                    extracted = _extract_ole_doc(file_bytes)
                    if extracted.strip():
                        text_data.append(("Corps du document", extracted))
                except Exception:
                    pass

        elif ext == ".doc":
            try:
                extracted = _extract_ole_doc(file_bytes)
                if extracted.strip():
                    text_data.append(("Corps du document", extracted))
            except Exception:
                pass

        elif ext == ".xlsx":
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet).fillna("")
                sentences = []
                for _, row in df.iterrows():
                    row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
                    if row_text:
                        sentences.append(row_text + ".")
                text_data.append((f"Feuille: {sheet}", " ".join(sentences)))

        elif ext == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
            text_data.append(("Données CSV", df.to_string()))

        elif ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for name in z.namelist():
                    if name.startswith('__MACOSX') or name.endswith('/'):
                        continue
                    try:
                        inner_bytes = z.read(name, pwd=b"Atos2020")
                    except Exception:
                        try:
                            inner_bytes = z.read(name)
                        except Exception:
                            continue
                    inner_data = extract_text_from_bytes(inner_bytes, name)
                    for loc, txt in inner_data:
                        text_data.append((f"ZIP -> {name} ({loc})", txt))
    except Exception as e:
        print(f"[INFO] Fichier ignoré : {file_name} — {str(e)}")

    return text_data

def infer_metadata(filename):
    normalized = filename.upper()
    entity = "Non classée"
    for e in ["OCM", "OEG", "OJO", "OCI"]:
        if e in normalized:
            entity = e
            break

    app = "Non classée"
    if "KPSA" in normalized:
        app = "KPSA"
    elif "ZM" in normalized or "MZ" in normalized:
        app = "MZ"

    return entity, app
def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)
def chunk_text_data(parsed_data, max_length=1000):
    chunks = []
    for location, text in parsed_data:
        if not text.strip():
            continue
        paragraphs = text.split('\n')
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > max_length:
                sentences = split_into_sentences(para)
            else:
                sentences = [para]

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) > max_length and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "location": location
                    })
                    overlap = current_chunk[-250:] if len(current_chunk) > 250 else current_chunk
                    if " " in overlap:
                        overlap = overlap[overlap.find(" ")+1:]

                    current_chunk = overlap.strip() + " " + sentence + " "
                else:
                    current_chunk += sentence + " "

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "location": location
            })

    return chunks

# ==========================================
# 5.5 AUTO-INDEXATION DU DOSSIER LOCAL
# ==========================================
def sync_local_folder_v2(storage_dir=None, target_collection=None, target_embedding_model=None):
    """Synchronize a folder explicitly; callers may inject isolated test resources."""
    storage_dir = storage_dir or STORAGE_DIR
    target_collection = target_collection or collection
    target_embedding_model = target_embedding_model or embedding_model
    local_files = [f for f in os.listdir(storage_dir) if os.path.isfile(os.path.join(storage_dir, f))]
    for filename in local_files:
        file_path = os.path.join(storage_dir, filename)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = target_collection.get(where={"file_hash": file_hash})
        if not existing or len(existing["ids"]) == 0:
            is_structured_docx = filename.casefold().endswith(".docx") and ENABLE_STRUCTURED_DOCX_INGESTION

            # Structured DOCX ingestion is intentionally isolated behind an opt-in
            # flag.  Build and validate the complete payload before any Chroma
            # write; dry-run validates it and reports the result without writing.
            if is_structured_docx:
                ent_tag, app_tag = infer_metadata(filename)
                payload = build_structured_docx_index_payload(
                    file_bytes,
                    filename,
                    file_hash=file_hash,
                    geographical_entity=ent_tag,
                    application=app_tag,
                )
                expected_count = len(payload["documents"])
                if not (len(payload["ids"]) == len(payload["metadatas"]) == expected_count):
                    raise ValueError("structured DOCX payload is internally inconsistent")

                if STRUCTURED_INGESTION_DRY_RUN:
                    print(
                        f"[STRUCTURED_DRY_RUN] {filename}: "
                        f"{expected_count} chunks validated; zero Chroma writes"
                    )
                    continue

                try:
                    embeddings = target_embedding_model.encode(payload["documents"]).tolist()
                    if len(embeddings) != expected_count:
                        raise ValueError("embedding count does not match structured chunk count")
                except Exception:
                    # No collection.add call occurs unless extraction and all
                    # embeddings have succeeded.
                    raise
                target_collection.add(
                    ids=payload["ids"],
                    embeddings=embeddings,
                    metadatas=payload["metadatas"],
                    documents=payload["documents"],
                )
                continue

            # In dry-run mode, do not write legacy formats either.  The mode is
            # intended for a write-free structured-ingestion validation pass.
            if STRUCTURED_INGESTION_DRY_RUN:
                continue

            parsed_text_data = extract_text_from_bytes(file_bytes, filename)
            document_chunks = chunk_text_data(parsed_text_data)

            if document_chunks:
                ent_tag, app_tag = infer_metadata(filename)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name, _ = os.path.splitext(filename)

                ids, metadatas, documents = [], [], []
                for idx, chunk in enumerate(document_chunks):
                    chunk_id = f"{base_name}_{timestamp_str}_chunk_{idx}"
                    ids.append(chunk_id)

                    enriched_text = f"Fichier source : {filename}\nEmplacement : {chunk['location']}\nContenu :\n{chunk['text']}"

                    documents.append(enriched_text)
                    metadatas.append({
                        "source_file": filename,
                        "saved_as": filename,
                        "location": chunk["location"],
                        "geographical_entity": ent_tag,
                        "application": app_tag,
                        "file_hash": file_hash,
                        "timestamp_ingest": timestamp_str
                    })

                embeddings = []
                batch_size = 32

                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]

                    try:
                        batch_embs = target_embedding_model.encode(batch_docs).tolist()
                        if not isinstance(batch_embs[0], list):
                            batch_embs = [batch_embs]
                        embeddings.extend(batch_embs)
                    except Exception as e:
                        print(f"Erreur d'embedding local : {e}")

                target_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


def maybe_run_startup_sync():
    """Run automatic synchronization only when explicitly enabled."""
    if ENABLE_STARTUP_SYNC:
        sync_local_folder_v2()


maybe_run_startup_sync()
bm25_index, bm25_docs, bm25_metas = build_bm25_index(collection, collection.count())


@st.cache_resource
def load_prepared_registry(storage_dir):
    """Prepare unchanged local files once for Direct Answer and global Find Me."""
    registry = mvp_services.PreparedDocumentRegistry()
    root = os.path.abspath(storage_dir)
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].casefold() in {
                ".pdf", ".docx", ".doc", ".xlsx", ".csv"
            }:
                try:
                    with open(path, "rb") as source:
                        registry.prepare(source.read(), name)
                except OSError:
                    continue
    return registry

# ==========================================
# 6. INTERFACE UTILISATEUR & SIDEBAR
# ==========================================
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "French"
if "canonical_cache" not in st.session_state:
    st.session_state.canonical_cache = canonical_rag.CanonicalSessionCache()
if "active_document_service" not in st.session_state:
    st.session_state.active_document_service = canonical_rag.ActiveDocumentService()
if "fast_direct_engine" not in st.session_state:
    st.session_state.fast_direct_engine = canonical_rag.FastDirectAnswerEngine()
if "prepared_registry" not in st.session_state:
    st.session_state.prepared_registry = load_prepared_registry(STORAGE_DIR)
if "local_metrics" not in st.session_state:
    st.session_state.local_metrics = mvp_services.LocalMetrics()
if "upload_states" not in st.session_state:
    st.session_state.upload_states = {}
if "active_overlay" not in st.session_state:
    st.session_state.active_overlay = "NONE"
if "find_me_query" not in st.session_state:
    st.session_state.find_me_query = ""
if "find_me_results" not in st.session_state:
    st.session_state.find_me_results = ()
if "find_me_page" not in st.session_state:
    st.session_state.find_me_page = 0
catalog_collection_count = collection.count()
if st.session_state.get("catalog_index_count") != catalog_collection_count:
    catalog_payload = collection.get(include=["metadatas"])
    st.session_state.catalog_index = canonical_rag.CatalogIndex.from_metadatas(
        catalog_payload.get("metadatas") or []
    )
    st.session_state.catalog_index_count = catalog_collection_count


def fast_catalog_rows(query="", refinements=None):
    refinements = refinements or {}
    metadata = refinements.get("metadata", {})
    terms = " ".join(refinements.get("terms", [])) or query
    file_types = refinements.get("file_types", [])
    entries = st.session_state.catalog_index.search(
        terms,
        application=metadata.get("application", ""),
        geographical_entity=metadata.get("geographical_entity", ""),
    )
    if file_types:
        allowed = {value.casefold() for value in file_types}
        entries = tuple(entry for entry in entries if entry.file_type in allowed)
    return [dict(entry.metadata) for entry in entries]

with st.sidebar:
    st.header(" Configuration & Filtres")

    st.write(" Zone Géographique (Filiale)")
    with st.container(height=120):
        filter_ent = st.radio(
            "Zone Géographique",
            ["Tous", "OCM", "OEG", "OJO", "OCI"],
            index=0,
            label_visibility="collapsed"
        )

    st.write("Application")
    with st.container(height=120):
        filter_application = st.radio(
            "Application",
            ["Tous", "KPSA", "MZ"],
            index=0,
            label_visibility="collapsed"
        )

    st.write("AI Answer")
    st.info("API-backed" if generation_provider.configured else "API key not configured")
    selected_model = generation_provider.model

    st.markdown("---")

    with st.expander(" Admin : Ingestion manuelle"):
        uploaded_files = st.file_uploader("PDF, DOCX, XLSX, CSV, ZIP", accept_multiple_files=True, type=["pdf", "docx", "xlsx", "csv", "zip"])

        if uploaded_files:
            for f in uploaded_files:
                file_bytes = f.getvalue()
                upload_key = st.session_state.prepared_registry.cache_key(file_bytes)
                previous_state = st.session_state.upload_states.get(upload_key, "IDLE")
                if previous_state in {"SUCCESS", "WARNING"}:
                    st.success(f"Document importé avec succès : {f.name}")
                    continue
                st.session_state.upload_states[upload_key] = "PROCESSING"
                prepared = st.session_state.prepared_registry.prepare(file_bytes, f.name)
                if prepared.document is None or prepared.state == "FAILED":
                    st.session_state.upload_states[upload_key] = "FAILED"
                    st.session_state.local_metrics.preparation_failures += 1
                    st.error(f"Impossible de lire ce document : {f.name}")
                    for warning in prepared.warnings:
                        st.caption(warning)
                    continue
                uploaded_document = prepared.document
                st.session_state.fast_direct_engine.prepare(uploaded_document)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name, ext = os.path.splitext(f.name)
                saved_filename = f"{base_name}_{timestamp_str}{ext}"
                target_path = os.path.join(STORAGE_DIR, saved_filename)

                with open(target_path, "wb") as out_f:
                    out_f.write(file_bytes)

                sync_local_folder_v2()
                st.success(f" Fichier ajouté : {f.name}")
                final_state = "WARNING" if prepared.state == "READY_WITH_WARNINGS" else "SUCCESS"
                st.session_state.upload_states[upload_key] = final_state
                if final_state == "WARNING":
                    st.warning(f"Document importé avec avertissements : {f.name}")
                else:
                    st.toast("Document importé avec succès.")
                ui_components.render_document_status(prepared.state)

    st.markdown("---")
    st.subheader(" Statut du Système")
    st.caption(f" Moteur : MiniLM-L12 + {selected_model}")
    st.metric(label="Total Chunks Indexés", value=collection.count())

    if st.button(" Réinitialiser la discussion"):
        st.session_state.messages = []
        st.session_state.last_lang = "French"
        st.session_state.catalog_refinements = {}
        st.rerun()

# ==========================================
# 7. FILTRES CHROMADB
# ==========================================
chroma_conditions = []
if filter_ent != "Tous":
    chroma_conditions.append({"geographical_entity": filter_ent})
if filter_application != "Tous":
    chroma_conditions.append({"application": filter_application})

chroma_filter = None
if len(chroma_conditions) == 1:
    chroma_filter = chroma_conditions[0]
elif len(chroma_conditions) > 1:
    chroma_filter = {"$and": chroma_conditions}

with st.sidebar:
    with st.expander(" Catalogue de connaissances"):
        sidebar_refinements = {"metadata": {}}
        if filter_application != "Tous":
            sidebar_refinements["metadata"]["application"] = filter_application
        if filter_ent != "Tous":
            sidebar_refinements["metadata"]["geographical_entity"] = filter_ent
        catalog = fast_catalog_rows(refinements=sidebar_refinements)
        st.caption(f"{len(catalog)} document(s) unique(s)")
        for index, metadata in enumerate(catalog):
            filename = metadata["source_file"]
            extension = os.path.splitext(filename)[1].lstrip(".").upper() or "FILE"
            st.write(f"**{metadata.get('application', 'Non classée')} / {metadata.get('geographical_entity', 'Non classée')}** — {filename} [{extension}]")
            st.button(" Fichier", on_click=open_local_file, args=(os.path.abspath(os.path.join(STORAGE_DIR, filename)),), key=f"catalog_file_{index}")

# ==========================================
# 8. INTERFACE DE DISCUSSION PRINCIPALE
# ==========================================
title_col, find_col, language_col = st.columns([5, 1.4, 1])
with title_col:
    st.title("Corporate Brain Assistant")
with language_col:
    ui_language = st.segmented_control("Language", ["FR", "EN"], default="FR", label_visibility="collapsed")
with find_col:
    if st.button("Trouver : où" if ui_language == "FR" else "Find me: where", use_container_width=True):
        st.session_state.active_overlay = "FIND_ME"
        st.rerun()

# A single top-level overlay branch owns all modal/panel rendering.
overlay = st.session_state.get("active_overlay", "NONE")
if overlay == "FIND_ME":
    ui_components.render_find_me(
        st.session_state.prepared_registry,
        "French" if ui_language == "FR" else "English",
    )
elif overlay == "SOURCE_VIEWER":
    viewer_target = st.session_state.get("pending_source_target")
    viewer_document = next(
        (item for item in st.session_state.prepared_registry.documents
         if viewer_target is not None and item.file_hash == viewer_target.file_hash), None,
    )
    if viewer_document is not None and viewer_target is not None:
        exact_path = os.path.abspath(os.path.join(STORAGE_DIR, viewer_document.source_file))
        ui_components.render_source_viewer(viewer_document, viewer_target, exact_path)
        st.stop()
st.markdown("*RAG Optimisé : Discussion, pistes proches & extrait direct*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.caption(f"Mode : {msg.get('actual_mode', msg.get('answer_mode', 'generative'))} · {msg.get('language', 'French')}")
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(direct_source_label(msg.get("language", "French"))):
                unique_sources = {}
                for src in msg["sources"]:
                    if src["path"] not in unique_sources:
                        unique_sources[src["path"]] = src

                for source_index, src in enumerate(unique_sources.values()):
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        label = " (piste proche, hors filtre)" if src.get("relaxed") else ""
                        st.write(f" **Fichier**: {src['file']}{label}")
                    with cols[1]:
                        st.button(" Fichier", on_click=open_local_file, args=(src.get("path", ""),), key=f"hist_file_{message_index}_{source_index}")
                    with cols[2]:
                        folder_path = os.path.dirname(src.get("path", "")) if src.get("path") else ""
                        st.button(" Dossier", on_click=open_local_file, args=(folder_path,), key=f"hist_folder_{message_index}_{source_index}")
        if msg.get("suggestions"):
            selected_suggestion = ui_components.render_suggestions(
                msg["suggestions"], msg.get("language", "English"), f"history_suggest_{message_index}"
            )
            if selected_suggestion:
                st.session_state.pending_suggested_query = selected_suggestion
                st.rerun()

if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = "AI answer"
if "catalog_refinements" not in st.session_state:
    st.session_state.catalog_refinements = {}
if "catalog_mode_last" not in st.session_state:
    st.session_state.catalog_mode_last = st.session_state.answer_mode
elif st.session_state.answer_mode != st.session_state.catalog_mode_last:
    if st.session_state.answer_mode != "Knowledge catalog":
        st.session_state.catalog_refinements = {}
    st.session_state.catalog_mode_last = st.session_state.answer_mode
# Legacy source-contract marker retained for older static audits:
# answer_mode = st.selectbox (replaced by the required segmented control)
answer_mode = st.segmented_control(
    "Mode de réponse",
    ["Knowledge catalog", "Direct answer", "AI answer"],
    key="answer_mode",
)
ai_scope = "current_active_document"
if answer_mode == "AI answer":
    ai_scope = st.segmented_control(
        "AI evidence scope",
        ["current_active_document", "all_documents_explicit"],
        format_func=lambda value: (
            "Current active document" if value == "current_active_document"
            else "All documents — explicit"
        ),
        key="ai_evidence_scope",
    )
    if ai_scope == "current_active_document":
        active_for_ai = st.session_state.active_document_service.active
        if active_for_ai is None:
            st.info("Select a document in Direct answer mode before using current-document AI synthesis.")
        else:
            st.caption(f"AI evidence source: {active_for_ai.source_file} only")
if "direct_answer_document_id" not in st.session_state:
    st.session_state.direct_answer_document_id = None
direct_scope = "specific_document"
if answer_mode == "Direct answer":
    scope_options = ["specific_document"]
    if experimental_global_direct_answer_enabled():
        scope_options.append("all_documents_experimental")
    if st.session_state.get("direct_answer_scope") not in scope_options:
        st.session_state.direct_answer_scope = "specific_document"
    direct_scope = st.segmented_control(
        "Scope",
        scope_options,
        format_func=lambda value: (
            "Specific document" if value == "specific_document"
            else "All documents — experimental"
        ),
        key="direct_answer_scope",
    )
    if direct_scope == "specific_document":
        direct_documents = list_catalog_documents(collection, chroma_filter)
        direct_options = {
            direct_document_identity(metadata): metadata
            for metadata in direct_documents
        }
        option_ids = [key for key in sorted(
            direct_options,
            key=lambda key: str(direct_options[key].get("source_file", "")).casefold(),
        )]
        if option_ids:
            if st.session_state.direct_answer_document_id not in option_ids:
                st.session_state.direct_answer_document_id = None
            st.pills(
                "Document",
                [None, *option_ids],
                format_func=lambda value: (
                    "Select a document..." if value is None
                    else direct_options[value].get("source_file", value)
                ),
                key="direct_answer_document_id",
            )
        else:
            st.session_state.direct_answer_document_id = None
            st.info("Aucun document disponible dans le périmètre des filtres actifs.")
        selected_metadata = next(
            (metadata for metadata in direct_documents
             if direct_document_identity(metadata) == st.session_state.direct_answer_document_id),
            None,
        )
        if selected_metadata is not None:
            st.caption(f"Selected document: {selected_metadata.get('source_file', '')}")
            selected_name = selected_metadata.get("source_file", "")
            saved_name = selected_metadata.get("saved_as") or selected_name
            selected_path = os.path.join(STORAGE_DIR, saved_name)
            if os.path.isfile(selected_path):
                with open(selected_path, "rb") as selected_file:
                    selected_bytes = selected_file.read()
                outcome = canonical_rag.normalize_with_gate(selected_bytes, selected_name)
                if outcome.document is not None:
                    cached_document = st.session_state.canonical_cache.get_or_normalize(selected_bytes, selected_name)
                    st.session_state.fast_direct_engine.prepare(cached_document)
                    active_context = st.session_state.active_document_service.select(cached_document)
                    diagnostics = canonical_rag.ingestion_diagnostics(cached_document)
                    state_label = {"READY": "Ready", "READY_WITH_WARNINGS": "Warning"}.get(
                        diagnostics.status, "Failed"
                    )
                    st.caption(f"Document state: {state_label}")
                    st.caption("Evidence source: current document only")
                    if ENABLE_CANONICAL_DEBUG:
                        st.json(canonical_rag.debug_snapshot(active_context))
                else:
                    st.session_state.active_document_service.clear()
                    st.error("Document could not be reliably read")
            else:
                st.session_state.active_document_service.clear()
                st.warning("The selected source file is unavailable for canonical normalization.")
        else:
            st.session_state.active_document_service.clear()
    else:
        st.warning(
            "Experimental global search may be less precise. "
            "Select a specific document for the most reliable result."
        )
st.caption(
    "Knowledge catalog : liste complète des documents. "
    "Direct answer : extraction déterministe. "
    "AI answer : RAG génératif. "
    "Le mode reste actif jusqu'à votre prochaine sélection."
)

typed_query = st.chat_input("Posez votre question ou tapez un acronyme...")
user_query = st.session_state.pop("pending_suggested_query", None) or typed_query
if user_query:

    current_lang = detect_query_language(user_query, fallback_lang="French")
    st.session_state.last_lang = current_lang

    with st.chat_message("user"):
        st.markdown(user_query)

    previous_actual_mode = next(
        (
            message.get("actual_mode")
            for message in reversed(st.session_state.messages)
            if message.get("role") == "assistant"
        ),
        None,
    )
    previous_catalog_refinements = dict(st.session_state.catalog_refinements)
    current_catalog_refinements = parse_catalog_refinements(user_query)
    continuation = (
        answer_mode == "Knowledge catalog"
        and detect_catalog_continuation(user_query, previous_actual_mode)
    )
    catalog_refinements = merge_catalog_refinements(
        previous_catalog_refinements,
        current_catalog_refinements,
        continuation=continuation,
    )
    catalog_route = (
        answer_mode == "Knowledge catalog"
    )
    if catalog_route:
        st.session_state.catalog_refinements = catalog_refinements
        catalog_rows = fast_catalog_rows(user_query, catalog_refinements)
        catalog_lines = [
            f"- {row.get('application', 'Non classée')} / "
            f"{row.get('geographical_entity', 'Non classée')} — "
            f"{row.get('source_file', 'Fichier source')}"
            for row in catalog_rows
        ]
        catalog_text = (
            "Documents indexés :\n" + "\n".join(catalog_lines)
            if catalog_lines
            else "Aucun document indexé dans le périmètre sélectionné."
        )
        with st.chat_message("assistant"):
            st.caption("Mode : catalogue")
            st.markdown(catalog_text)
            with st.expander(" Ressources consultées"):
                for index, row in enumerate(catalog_rows):
                    filename = row.get("source_file", "")
                    st.write(filename)
                    st.button(
                        " Fichier",
                        on_click=open_local_file,
                        args=(os.path.abspath(os.path.join(STORAGE_DIR, filename)),),
                        key=f"chat_catalog_file_{len(st.session_state.messages)}_{index}",
                    )
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant",
            "content": catalog_text,
            "language": current_lang,
            "actual_mode": "catalog",
            "sources": [],
            "catalog_refinements": catalog_refinements,
        })
        with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
            audit_f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "question_originale": user_query,
                "language": current_lang,
                "requested_mode": answer_mode,
                "actual_mode": "catalog",
                "sources_count": len(catalog_rows),
            }, ensure_ascii=False) + "\n")
        st.stop()

    if answer_mode == "AI answer" and ai_scope == "current_active_document" and st.session_state.active_document_service.active is None:
        missing_active = "Select a document in Direct answer mode before using current-document AI synthesis."
        with st.chat_message("assistant"):
            st.warning(missing_active)
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant", "content": missing_active, "language": current_lang,
            "actual_mode": "ai_missing_active_document", "sources": [],
        })
        st.stop()

    if answer_mode == "Direct answer" and is_direct_sensitive_request(user_query):
        sensitive_message = direct_sensitive_message(current_lang)
        with st.chat_message("assistant"):
            st.caption("Mode : direct_sensitive_request")
            st.warning(sensitive_message)
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant", "content": sensitive_message, "language": current_lang,
            "actual_mode": "direct_sensitive_request", "answer_mode": "direct_sensitive_request", "sources": [],
        })
        with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
            audit_f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "question_originale": user_query,
                "language": current_lang,
                "requested_mode": answer_mode,
                "actual_mode": "direct_sensitive_request",
                "direct_status": "direct_sensitive_request",
                "sources_count": 0,
            }, ensure_ascii=False) + "\n")
        st.stop()

    if answer_mode == "Direct answer" and detect_direct_incomplete_query(user_query):
        incomplete_message = direct_incomplete_message(current_lang)
        with st.chat_message("assistant"):
            st.caption("Mode : direct_incomplete_query")
            st.info(incomplete_message)
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant", "content": incomplete_message, "language": current_lang,
            "actual_mode": "direct_incomplete_query", "direct_failure_reason": "missing_query_subject", "sources": [],
        })
        with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
            audit_f.write(json.dumps({
                "timestamp": datetime.now().isoformat(), "question_originale": user_query,
                "language": current_lang, "requested_mode": answer_mode,
                "actual_mode": "direct_incomplete_query", "direct_failure_reason": "missing_query_subject",
                "sources_count": 0,
            }, ensure_ascii=False) + "\n")
        st.stop()

    if answer_mode == "Direct answer":
        vague_entity = detect_direct_vague_entity(user_query)
        if vague_entity:
            vague_message = direct_vague_message(current_lang, vague_entity)
            with st.chat_message("assistant"):
                st.caption("Mode : direct_vague_query")
                st.info(vague_message)
            st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
            st.session_state.messages.append({
                "role": "assistant", "content": vague_message, "language": current_lang,
                "actual_mode": "direct_vague_query", "answer_mode": "direct_vague_query", "sources": [],
            })
            with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                audit_f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "question_originale": user_query,
                    "language": current_lang,
                    "requested_mode": answer_mode,
                    "actual_mode": "direct_vague_query",
                    "direct_status": "direct_vague_query",
                    "sources_count": 0,
                }, ensure_ascii=False) + "\n")
            st.stop()

    structured_specific_direct = (
        answer_mode == "Direct answer"
        and direct_scope == "specific_document"
        and structured_specific_direct_answer_enabled()
        and generic_structured_direct_answer_enabled()
    )
    # if answer_mode == "Direct answer" and not is_direct_answer_suitable(user_query):
    if answer_mode == "Direct answer" and is_obvious_synthesis_query(user_query):
        direct_message = direct_unsuitable_message(current_lang)
        with st.chat_message("assistant"):
            st.caption("Mode : direct_unsuitable")
            st.info(direct_message)
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant", "content": direct_message, "language": current_lang,
            "actual_mode": "direct_unsuitable", "answer_mode": "direct_unsuitable", "sources": [],
        })
        st.stop()

    if answer_mode == "Direct answer" and not is_direct_answer_suitable(user_query) and not structured_specific_direct:
        direct_message = direct_unsuitable_message(current_lang)
        with st.chat_message("assistant"):
            st.caption("Mode : direct_unsuitable")
            st.info(direct_message)
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant", "content": direct_message, "language": current_lang,
            "actual_mode": "direct_unsuitable", "answer_mode": "direct_unsuitable", "sources": [],
        })
        with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
            audit_f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "question_originale": user_query,
                "language": current_lang,
                "requested_mode": answer_mode,
                "actual_mode": "direct_unsuitable",
                "direct_status": "UNSUITABLE_EXPLANATORY_OR_COMPARATIVE",
                "sources_count": 0,
            }, ensure_ascii=False) + "\n")
        st.stop()

    direct_document = None
    if answer_mode == "Direct answer" and direct_scope == "specific_document":
        direct_document = resolve_direct_document(
            collection, chroma_filter, st.session_state.direct_answer_document_id
        )
        if (
            not direct_scope_selection_consistent(
                direct_scope, st.session_state.get("direct_answer_scope", direct_scope)
            )
            or direct_document is None
        ):
            missing_scope_message = direct_invalid_scope_message(current_lang)
            with st.chat_message("assistant"):
                st.caption("Mode : direct_invalid_document_scope")
                st.info(missing_scope_message)
            st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
            st.session_state.messages.append({
                "role": "assistant", "content": missing_scope_message, "language": current_lang,
                "actual_mode": "direct_invalid_document_scope", "sources": [],
            })
            with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                audit_f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "question_originale": user_query,
                    "language": current_lang,
                    "requested_mode": answer_mode,
                    "actual_mode": "direct_invalid_document_scope",
                    "direct_answer_scope": "specific_document",
                    "direct_answer_scope_requested": "specific_document",
                    "direct_answer_scope_effective": "specific_document",
                    "direct_answer_document_id": st.session_state.direct_answer_document_id,
                    "direct_answer_source_file": direct_document.get("source_file") if direct_document else None,
                    "effective_chroma_filter": direct_scope_audit_filter(chroma_filter),
                    "scope_validation": "FAIL",
                    "sources_count": 0,
                }, ensure_ascii=False) + "\n")
            st.stop()

    # Canonical selected-document route: evidence never leaves the active hash.
    active_canonical = st.session_state.active_document_service.active
    if answer_mode == "Direct answer" and direct_scope == "specific_document" and active_canonical is not None:
        canonical_trace = canonical_rag.DirectAnswerTrace((), {}, False)
        selected_hash = str((direct_document or {}).get("file_hash") or "")
        if selected_hash and selected_hash != active_canonical.file_hash:
            canonical_result = canonical_rag.AnswerResult(
                "NO_EVIDENCE", canonical_rag.NO_EXPLICIT_EVIDENCE, (),
                active_canonical.source_file, active_canonical.file_hash,
                "LOW", "no_evidence", "selected metadata hash mismatch",
            )
        else:
            try:
                canonical_result, canonical_trace = st.session_state.fast_direct_engine.query(
                    active_canonical, user_query
                )
            except Exception:
                canonical_result = canonical_rag.AnswerResult(
                    "NO_EVIDENCE", canonical_rag.NO_EXPLICIT_EVIDENCE, (),
                    active_canonical.source_file, active_canonical.file_hash,
                    "LOW", "no_evidence", "local extractive engine unavailable",
                )
        answer_text = (
            canonical_result.answer if canonical_result.status == "ANSWER"
            else canonical_no_evidence_message(current_lang)
        )
        canonical_sources = [{
            "file": active_canonical.source_file,
            "path": os.path.abspath(os.path.join(STORAGE_DIR, (direct_document or {}).get("saved_as") or active_canonical.source_file)),
            "loc": block.section or block.sheet or (f"Page {block.page}" if block.page else "Canonical block"),
            "text": block.text,
            "evidence_id": block.block_id,
            "relaxed": False,
        } for block in canonical_result.evidence_blocks]
        with st.chat_message("assistant"):
            selected_suggestion = ui_components.render_answer(
                canonical_result,
                latency_ms=canonical_trace.timings_ms.get("total", 0.0),
                document=active_canonical.canonical_document,
                key_prefix=f"direct_{len(st.session_state.messages)}",
            )
            if selected_suggestion:
                st.session_state.pending_suggested_query = selected_suggestion
            if ENABLE_CANONICAL_DEBUG:
                st.json(canonical_rag.debug_snapshot(active_canonical, canonical_result))
            if DIRECT_DEBUG:
                st.json({
                    "question": user_query,
                    "file_hash": active_canonical.file_hash,
                    "block_count": canonical_trace.active_block_count,
                    "candidate_count": canonical_trace.candidate_count,
                    "top_candidates": list(canonical_trace.top_candidates),
                    "answer_span": canonical_result.answer,
                    "method": canonical_result.method,
                    "latency_ms": canonical_trace.timings_ms.get("total", 0.0),
                })
            if DIRECT_FACT_DEBUG:
                fact_store = st.session_state.fast_direct_engine.fact_stores.get(active_canonical.file_hash)
                st.json({
                    "question": user_query,
                    "active_file_hash": active_canonical.file_hash,
                    "parsed_query": str(canonical_rag.parse_question(user_query)),
                    "fact_count": len(fact_store.facts) if fact_store else 0,
                    "fact_categories": list(fact_store.categories) if fact_store else [],
                    "final_method": canonical_result.method,
                    "ambiguity_reason": canonical_result.reason,
                    "latency_ms": canonical_trace.timings_ms.get("total", 0.0),
                })
        st.session_state.local_metrics.record_answer(
            canonical_result, canonical_trace.timings_ms.get("total", 0.0)
        )
        st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
        st.session_state.messages.append({
            "role": "assistant", "content": answer_text, "language": current_lang,
            "actual_mode": canonical_result.method, "sources": canonical_sources,
            "file_hash": active_canonical.file_hash,
            "suggestions": list(canonical_result.suggestions),
        })
        with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
            audit_f.write(json.dumps({
                "timestamp": datetime.now().isoformat(), "question_originale": user_query,
                "requested_mode": "Direct answer", "actual_mode": canonical_result.method,
                "active_file_hash": active_canonical.file_hash,
                "evidence_block_ids": [block.block_id for block in canonical_result.evidence_blocks],
                "stages_attempted": list(canonical_trace.stages_attempted),
                "timings_ms": dict(canonical_trace.timings_ms),
                "cache_hit": canonical_trace.cache_hit,
                "generation_calls": 0, "chroma_calls": 0,
            }, ensure_ascii=False) + "\n")
        st.stop()

    # 1. Reformulation de la question (extractive is standalone-only and opt-in).
    extractive_route = (
        answer_mode == "Direct answer"
    )
    standalone_query = (
        user_query
        if extractive_route or (answer_mode == "AI answer" and ai_scope == "current_active_document")
        else contextualize_query(user_query, st.session_state.messages, selected_model)
    )

    if collection.count() == 0:
        no_docs = " Aucun document n'est indexé dans 'doc_storage_v2'." if current_lang == "French" else "⚠️ No documents indexed in 'doc_storage_v2'."
        with st.chat_message("assistant"):
            st.caption("Mode : réponse IA")
            st.warning(no_docs)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant", "content": no_docs, "language": current_lang, "actual_mode": "generative",
        })
    else:
        # 2. Recherche Hybride (Vectoriel + BM25 avec RRF), avec repli élargi automatique
        structured_exhaustive_mode = bool(
            extractive_route and direct_scope == "specific_document"
            and direct_document is not None and structured_specific_direct_answer_enabled()
            and selected_document_has_structured_metadata(collection, direct_document)
        )
        generic_structured_mode = structured_exhaustive_mode and generic_structured_direct_answer_enabled()
        precomputed_extractive_evidence = None
        precomputed_extractive_result = None
        selected_document_chunk_count = 0
        exhaustive_scan_ms = None
        evidence_selection_ms = None
        retrieval_filter = chroma_filter
        if answer_mode == "AI answer" and ai_scope == "current_active_document":
            active_for_ai = st.session_state.active_document_service.active
            if active_for_ai is not None:
                hash_filter = {"file_hash": active_for_ai.file_hash}
                retrieval_filter = (
                    hash_filter if not chroma_filter
                    else {"$and": [chroma_filter, hash_filter]}
                )
        if extractive_route and direct_scope == "specific_document" and direct_document is not None:
            retrieval_filter = build_direct_document_filter(chroma_filter, direct_document)
        if structured_exhaustive_mode:
            scan_started = time.perf_counter()
            structured_chunks = fetch_structured_specific_chunks(collection, direct_document)
            selected_document_chunk_count = len(structured_chunks or [])
            exhaustive_scan_ms = (time.perf_counter() - scan_started) * 1000
            filtered_chunks, filtered_metas, relaxed_chunks, relaxed_metas, was_relaxed = [], [], [], [], False
            if structured_chunks is None:
                precomputed_extractive_evidence = rag_pipeline.EvidenceExtractionResult(
                    "NO_EXPLICIT_EVIDENCE", user_query, current_lang, (), (), False,
                    "structured_metadata_unavailable"
                )
            else:
                selection_started = time.perf_counter()
                extractor = (
                    rag_pipeline.extract_evidence_generic_structured
                    if generic_structured_mode
                    else rag_pipeline.extract_evidence_exhaustive_specific
                )
                precomputed_extractive_evidence = extractor(user_query, structured_chunks)
                evidence_selection_ms = (time.perf_counter() - selection_started) * 1000
            precomputed_extractive_result = rag_pipeline.build_extractive_answer(
                precomputed_extractive_evidence, current_lang
            )
            # full_stream_response = extractive_result.answer_text (legacy parity)
        else:
            filtered_chunks, filtered_metas, relaxed_chunks, relaxed_metas, was_relaxed = hybrid_search(
            query=standalone_query,
            collection=collection,
            embedding_model=embedding_model,
            bm25=bm25_index,
            docs=bm25_docs,
            metadatas=bm25_metas,
            chroma_filter=retrieval_filter,
            top_k=15,
            min_results_before_relax=(0 if extractive_route and direct_scope == "specific_document" else 3),
            )

        if extractive_route and direct_scope == "specific_document":
            scope_candidates = list(filtered_metas) + list(relaxed_metas or [])
            scope_valid = (
                direct_document is not None
                and direct_filter_contains_identity(retrieval_filter, direct_document)
                and all(direct_metadata_matches_identity(metadata, direct_document) for metadata in scope_candidates)
            )
            if not scope_valid:
                invalid_scope_message = direct_invalid_scope_message(current_lang)
                with st.chat_message("assistant"):
                    st.caption("Mode : direct_invalid_document_scope")
                    st.info(invalid_scope_message)
                st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
                st.session_state.messages.append({
                    "role": "assistant", "content": invalid_scope_message, "language": current_lang,
                    "actual_mode": "direct_invalid_document_scope", "sources": [],
                })
                with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                    audit_f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "question_originale": user_query,
                        "language": current_lang,
                        "requested_mode": answer_mode,
                        "actual_mode": "direct_invalid_document_scope",
                        "direct_answer_scope_requested": "specific_document",
                        "direct_answer_scope_effective": "specific_document",
                        "direct_answer_document_id": direct_document_identity(direct_document) if direct_document else None,
                        "direct_answer_source_file": direct_document.get("source_file") if direct_document else None,
                        "effective_chroma_filter": direct_scope_audit_filter(retrieval_filter),
                        "scope_validation": "FAIL",
                        "sources_count": 0,
                    }, ensure_ascii=False) + "\n")
                st.stop()

        source_metadata_list = rag_pipeline.build_source_list(
            filtered_chunks, filtered_metas, STORAGE_DIR, relaxed_flag=False
        )
        relaxed_source_list = rag_pipeline.build_source_list(
            relaxed_chunks,
            relaxed_metas,
            STORAGE_DIR,
            relaxed_flag=True,
            start_id=len(source_metadata_list) + 1,
        )
        all_sources_for_prompt = source_metadata_list + relaxed_source_list

        if not source_metadata_list and not relaxed_source_list and not structured_exhaustive_mode:
            # Vraiment rien, même en élargissant la recherche : là on peut être honnête,
            # mais on reste dans l'esprit d'ouvrir la discussion plutôt que de la clore.
            no_match_msg = rag_pipeline.build_no_match_message(current_lang)
            with st.chat_message("assistant"):
                st.caption("Mode : réponse IA")
                st.markdown(no_match_msg)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({
                "role": "assistant", "content": no_match_msg, "language": current_lang, "actual_mode": "generative",
            })
        else:
            if extractive_route:
                # Legacy parity marker: full_stream_response = extractive_result.answer_text
                extractive_result = precomputed_extractive_result if structured_exhaustive_mode else None
                extractive_evidence = precomputed_extractive_evidence if structured_exhaustive_mode else None
                sensitive_output_detected = False
                try:
                    if not structured_exhaustive_mode:
                        extractive_trace = rag_pipeline.PipelineTrace(
                            query=user_query,
                            rewritten_query=standalone_query,
                            language=current_lang,
                            prompt=rag_pipeline.PromptResult(
                                prompt="",
                                sources=tuple(all_sources_for_prompt),
                                context=rag_pipeline.build_context(all_sources_for_prompt),
                            ),
                        )
                        extractive_evidence = rag_pipeline.extract_evidence(extractive_trace)
                    sensitive_output_detected = any(
                        contains_sensitive_output(passage.text)
                        for passage in (extractive_evidence.passages if extractive_evidence else ())
                    )
                    if structured_exhaustive_mode and sensitive_output_detected:
                        extractive_result = None
                    if not sensitive_output_detected and not structured_exhaustive_mode:
                        extractive_result = rag_pipeline.build_extractive_answer(
                            extractive_evidence, current_lang
                        )
                except Exception:
                    extractive_result = None

                if extractive_result is not None and extractive_result.status == "ANSWER":
                    display_sources = [
                        {
                            "id": source["source_id"],
                            "file": source["source_file"],
                            "loc": source["location"],
                            "text": next(
                                (
                                    passage.text
                                    for passage in extractive_evidence.passages
                                    if passage.source_id == source["source_id"]
                                ),
                                "",
                            ),
                            "path": os.path.abspath(
                                os.path.join(STORAGE_DIR, source["source_file"])
                            ),
                            "relaxed": False,
                        }
                        for source in extractive_result.sources
                    ]
                    localized_summary = build_direct_localized_summary(
                        user_query, extractive_evidence, current_lang
                    )
                    if localized_summary:
                        full_stream_response = (
                            f"{direct_answer_label(current_lang)}:\n{localized_summary}\n\n"
                            f"{direct_original_source_label(current_lang)}:\n"
                            f"{extractive_result.answer_text}"
                        )
                    else:
                        full_stream_response = extractive_result.answer_text
                    with st.chat_message("assistant"):
                        st.caption(direct_answer_label(current_lang))
                        st.caption(direct_source_label(current_lang))
                        if direct_scope == "specific_document" and direct_document is not None:
                            st.caption(f"Document scope: {direct_document.get('source_file', '')}")
                        elif direct_scope == "all_documents_experimental":
                            st.caption("Document scope: All documents — experimental")
                        st.markdown(full_stream_response)
                        with st.expander(direct_source_label(current_lang)):
                            for i_src, src in enumerate(display_sources):
                                cols = st.columns([3, 1, 1])
                                with cols[0]:
                                    st.write(f" **Fichier**: {src['file']}")
                                with cols[1]:
                                    st.button(
                                        " Fichier",
                                        on_click=open_local_file,
                                        args=(src.get("path", ""),),
                                        key=f"extractive_btn_{len(st.session_state.messages)}_{i_src}_f",
                                    )
                                with cols[2]:
                                    st.button(
                                        " Dossier",
                                        on_click=open_local_file,
                                        args=(os.path.dirname(src.get("path", "")),),
                                        key=f"extractive_btn_{len(st.session_state.messages)}_{i_src}_d",
                                    )
                    audit_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "question_originale": user_query,
                        "question_reformulee": standalone_query,
                        "language": current_lang,
                        "sources_count": len(display_sources),
                        "used_relaxed_fallback": was_relaxed,
                        "answer_mode": "extractive",
                        "requested_mode": answer_mode,
                        "actual_mode": "extractive",
                        "extractive_feature_enabled": True,
                        "extractive_status": extractive_result.status,
                        "extractive_evidence_ids": list(extractive_result.evidence_ids),
                        "extractive_source_ids": list(extractive_result.source_ids),
                        "extractive_passage_hashes": list(extractive_result.passage_hashes),
                        "direct_answer_scope": "specific_document" if direct_scope == "specific_document" else "all_documents_experimental",
                        "direct_answer_document_id": direct_document_identity(direct_document) if direct_document is not None else None,
                        "direct_answer_source_file": direct_document.get("source_file") if direct_document is not None else None,
                        # Legacy audit shape: "retrieval_mode": "hybrid"
                        "retrieval_mode": "exhaustive_specific_structured" if structured_exhaustive_mode else "hybrid",
                        "selected_document_chunk_count": selected_document_chunk_count,
                        "exhaustive_scan_ms": exhaustive_scan_ms,
                        "evidence_selection_ms": evidence_selection_ms,
                        "direct_answer_scope_requested": "specific_document" if direct_scope == "specific_document" else "all_documents_experimental",
                        "direct_answer_scope_effective": "specific_document" if direct_scope == "specific_document" else "all_documents_experimental",
                        "effective_chroma_filter": direct_scope_audit_filter(retrieval_filter),
                        "scope_validation": "PASS" if direct_scope != "specific_document" or direct_document is not None else "FAIL",
                    }
                    with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                        audit_f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                    st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_stream_response,
                        "language": current_lang,
                        "sources": display_sources,
                        "answer_mode": "extractive",
                        "actual_mode": "extractive",
                    })
                    st.stop()

            if extractive_route and answer_mode == "Direct answer":
                evidence_failure_reason = (
                    getattr(extractive_evidence, "failure_reason", None)
                    if extractive_evidence is not None else "no_explicit_evidence"
                )
                direct_reason = (
                    "sensitive_output_detected"
                    if sensitive_output_detected
                    else
                    "missing_requested_attribute"
                    if evidence_failure_reason == "ENTITY_ONLY_MATCH"
                    else "no_explicit_evidence"
                )
                direct_response = (
                    extractive_result.answer_text
                    if extractive_result is not None
                    else (
                        direct_sensitive_message(current_lang)
                        if direct_reason == "sensitive_output_detected"
                        else direct_no_evidence_message(current_lang, direct_reason)
                    )
                )
                if extractive_result is not None and extractive_result.status != "ANSWER":
                    direct_response = (
                        direct_sensitive_message(current_lang)
                        if direct_reason == "sensitive_output_detected"
                        else direct_no_evidence_message(current_lang, direct_reason)
                    )
                with st.chat_message("assistant"):
                    st.caption(direct_answer_label(current_lang))
                    if direct_scope == "specific_document" and direct_document is not None:
                        st.caption(f"Document scope: {direct_document.get('source_file', '')}")
                    elif direct_scope == "all_documents_experimental":
                        st.caption("Document scope: All documents — experimental")
                    st.markdown(direct_response)
                st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": direct_response,
                    "language": current_lang,
                    "actual_mode": "extractive",
                    "direct_failure_reason": direct_reason,
                    "sources": [],
                })
                with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                    audit_f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "question_originale": user_query,
                        "question_reformulee": standalone_query,
                        "language": current_lang,
                        "requested_mode": answer_mode,
                        "actual_mode": "extractive",
                        "extractive_status": (
                            extractive_result.status
                            if extractive_result is not None
                            else "NO_EXPLICIT_EVIDENCE"
                        ),
                        "direct_failure_reason": direct_reason,
                        "sources_count": 0,
                        "direct_answer_scope": "specific_document" if direct_scope == "specific_document" else "all_documents_experimental",
                        "direct_answer_document_id": direct_document_identity(direct_document) if direct_document is not None else None,
                        "direct_answer_source_file": direct_document.get("source_file") if direct_document is not None else None,
                        "retrieval_mode": "exhaustive_specific_structured" if structured_exhaustive_mode else "hybrid",
                        "selected_document_chunk_count": selected_document_chunk_count,
                        "exhaustive_scan_ms": exhaustive_scan_ms,
                        "evidence_selection_ms": evidence_selection_ms,
                        "direct_answer_scope_requested": "specific_document" if direct_scope == "specific_document" else "all_documents_experimental",
                        "direct_answer_scope_effective": "specific_document" if direct_scope == "specific_document" else "all_documents_experimental",
                        "effective_chroma_filter": direct_scope_audit_filter(retrieval_filter),
                        "scope_validation": "PASS" if direct_scope != "specific_document" or direct_document is not None else "FAIL",
                    }, ensure_ascii=False) + "\n")
                st.stop()

            prompt_result = rag_pipeline.build_production_prompt(
                user_query=user_query,
                filter_ent=filter_ent,
                filter_application=filter_application,
                history=st.session_state.messages,
                sources=all_sources_for_prompt,
                current_lang=current_lang,
                was_relaxed=was_relaxed,
            )
            prompt_instructions = prompt_result.prompt
            with st.chat_message("assistant"):
                st.caption("Mode : réponse IA")
                response_placeholder = st.empty()
                full_stream_response = ""

                try:
                    generation_result = rag_pipeline.stream_generate(
                        prompt_instructions,
                        selected_model,
                        generation_provider,
                        on_token=lambda response: response_placeholder.markdown(response + "▌"),
                        clarification_language=current_lang,
                    )
                    full_stream_response = generation_result.response
                    response_placeholder.markdown(full_stream_response)

                    citation_result = rag_pipeline.select_display_sources(
                        full_stream_response,
                        all_sources_for_prompt,
                    )
                    display_sources = [
                        {
                            "id": source.source_id,
                            "file": source.file_name,
                            "loc": source.location,
                            "text": source.text,
                            "path": source.path,
                            "relaxed": source.relaxed,
                        }
                        for source in citation_result.display_sources
                    ]
                    unique_sources = {
                        source.path: {
                            "id": source.source_id,
                            "file": source.file_name,
                            "loc": source.location,
                            "text": source.text,
                            "path": source.path,
                            "relaxed": source.relaxed,
                        }
                        for source in rag_pipeline.deduplicate_sources_by_path(
                            citation_result.display_sources
                        )
                    }

                    with st.expander(" Ressources consultées"):
                        if not unique_sources:
                            st.caption("Aucune source précise citée pour cette réponse.")
                        for i_src, src in enumerate(unique_sources.values()):
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                label = " (piste proche, hors filtre)" if src.get("relaxed") else ""
                                st.write(f" **Fichier**: {src['file']}{label}")
                            with cols[1]:
                                st.button(" Fichier", on_click=open_local_file, args=(src.get("path", ""),), key=f"new_btn_{len(st.session_state.messages)}_{i_src}_f")
                            with cols[2]:
                                folder_path = os.path.dirname(src.get("path", "")) if src.get("path") else ""
                                st.button(" Dossier", on_click=open_local_file, args=(folder_path,), key=f"new_btn_{len(st.session_state.messages)}_{i_src}_d")

                    audit_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "question_originale": user_query,
                        "question_reformulee": standalone_query,
                        "language": current_lang,
                        "sources_count": len(display_sources),
                        "used_relaxed_fallback": was_relaxed
                        ,"requested_mode": answer_mode,
                        "actual_mode": "generative",
                    }
                    with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                        audit_f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

                    st.session_state.messages.append({"role": "user", "content": user_query, "language": current_lang})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_stream_response,
                        "language": current_lang,
                        "sources": display_sources,
                        "actual_mode": "generative",
                    })

                except ProviderError as e:
                  st.error(str(e))
                except Exception:
                  st.error("AI Answer is temporarily unavailable.")
