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
from sentence_transformers import SentenceTransformer
import ollama
import docx2txt
import olefile
import struct
import unicodedata

import rag_pipeline

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
st.set_page_config(page_title="Corporate Brain - 100% Local (Qwen3)", layout="wide")

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

    # Fast multilingual model
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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


# ==========================================
# 4. FONCTIONS UTILITAIRES (DÉTECTION, MODELS, REFORMULATION)
# ==========================================

def detect_query_language(text, fallback_lang="French"):
    text_lower = text.lower().strip()

    en_keywords = {'what', 'how', 'why', 'who', 'where', 'when', 'is', 'are', 'the', 'this', 'that', 'can', 'you', 'explain', 'tell', 'me', 'show', 'list', 'details', 'about', 'english', 'mean', 'stand', 'eenglish'}
    fr_keywords = {'que', 'quoi', 'comment', 'pourquoi', 'qui', 'où', 'quand', 'est', 'sont', 'le', 'la', 'les', 'ce', 'cette', 'peux', 'tu', 'expliquer', 'montre', 'donne', 'veut', 'dire'}

    words = set(re.findall(r'\b\w+\b', text_lower))
    en_score = len(words.intersection(en_keywords))
    fr_score = len(words.intersection(fr_keywords))

    if en_score > fr_score:
        return "English"
    elif fr_score > en_score:
        return "French"
    else:
        return fallback_lang

def contextualize_query(user_query, chat_history, model_name):
    return rag_pipeline.rewrite_query(
        user_query,
        chat_history,
        model_name,
        ollama,
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
def sync_local_folder_v2():
    local_files = [f for f in os.listdir(STORAGE_DIR) if os.path.isfile(os.path.join(STORAGE_DIR, f))]
    for filename in local_files:
        file_path = os.path.join(STORAGE_DIR, filename)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = collection.get(where={"file_hash": file_hash})
        if not existing or len(existing["ids"]) == 0:
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
                        batch_embs = embedding_model.encode(batch_docs).tolist()
                        if not isinstance(batch_embs[0], list):
                            batch_embs = [batch_embs]
                        embeddings.extend(batch_embs)
                    except Exception as e:
                        print(f"Erreur d'embedding local : {e}")

                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

sync_local_folder_v2()
bm25_index, bm25_docs, bm25_metas = build_bm25_index(collection, collection.count())

# ==========================================
# 6. INTERFACE UTILISATEUR & SIDEBAR
# ==========================================
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "French"

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

    st.write("Modèle LLM")
    st.info("🧠 Qwen3:8b (Local)")
    selected_model = "qwen3:8b"

    st.markdown("---")

    with st.expander(" Admin : Ingestion manuelle"):
        uploaded_files = st.file_uploader("PDF, DOCX, XLSX, CSV, ZIP", accept_multiple_files=True, type=["pdf", "docx", "xlsx", "csv", "zip"])

        if uploaded_files:
            for f in uploaded_files:
                file_bytes = f.read()
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name, ext = os.path.splitext(f.name)
                saved_filename = f"{base_name}_{timestamp_str}{ext}"
                target_path = os.path.join(STORAGE_DIR, saved_filename)

                with open(target_path, "wb") as out_f:
                    out_f.write(file_bytes)

                sync_local_folder_v2()
                st.success(f" Fichier ajouté : {f.name}")
                st.rerun()

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
        catalog = list_catalog_documents(collection, chroma_filter)
        st.caption(f"{len(catalog)} document(s) unique(s)")
        for index, metadata in enumerate(catalog):
            filename = metadata["source_file"]
            extension = os.path.splitext(filename)[1].lstrip(".").upper() or "FILE"
            st.write(f"**{metadata.get('application', 'Non classée')} / {metadata.get('geographical_entity', 'Non classée')}** — {filename} [{extension}]")
            st.button(" Fichier", on_click=open_local_file, args=(os.path.abspath(os.path.join(STORAGE_DIR, filename)),), key=f"catalog_file_{index}")

# ==========================================
# 8. INTERFACE DE DISCUSSION PRINCIPALE
# ==========================================
st.title("🧠 Corporate Brain Assistant")
st.markdown("*RAG Optimisé : Discussion, pistes proches & extrait direct*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.caption(f"Mode : {msg.get('actual_mode', msg.get('answer_mode', 'generative'))}")
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(" Ressources consultées"):
                unique_sources = {}
                for src in msg["sources"]:
                    if src["path"] not in unique_sources:
                        unique_sources[src["path"]] = src

                for i_src, src in enumerate(unique_sources.values()):
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        label = " (piste proche, hors filtre)" if src.get("relaxed") else ""
                        st.write(f" **Fichier**: {src['file']}{label}")
                    with cols[1]:
                        st.button(" Fichier", on_click=open_local_file, args=(src.get("path", ""),), key=f"hist_btn_{st.session_state.messages.index(msg)}_{i_src}_f")
                    with cols[2]:
                        folder_path = os.path.dirname(src.get("path", "")) if src.get("path") else ""
                        st.button(" Dossier", on_click=open_local_file, args=(folder_path,), key=f"hist_btn_{st.session_state.messages.index(msg)}_{i_src}_d")

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
answer_mode = st.selectbox(
    "Mode de réponse",
    ["Knowledge catalog", "Direct answer", "AI answer"],
    key="answer_mode",
)
st.caption(
    "Knowledge catalog : liste complète des documents. "
    "Direct answer : extraction déterministe. "
    "AI answer : RAG génératif. "
    "Le mode reste actif jusqu'à votre prochaine sélection."
)

if user_query := st.chat_input("Posez votre question ou tapez un acronyme..."):

    current_lang = detect_query_language(user_query, fallback_lang=st.session_state.last_lang)
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
        catalog_rows = list_catalog_documents(
            collection, chroma_filter, catalog_refinements
        )
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
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant",
            "content": catalog_text,
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

    # 1. Reformulation de la question (extractive is standalone-only and opt-in).
    extractive_route = (
        answer_mode == "Direct answer"
    )
    standalone_query = (
        user_query
        if extractive_route
        else contextualize_query(user_query, st.session_state.messages, selected_model)
    )

    if collection.count() == 0:
        no_docs = " Aucun document n'est indexé dans 'doc_storage_v2'." if current_lang == "French" else "⚠️ No documents indexed in 'doc_storage_v2'."
        with st.chat_message("assistant"):
            st.caption("Mode : réponse IA")
            st.warning(no_docs)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant", "content": no_docs, "actual_mode": "generative",
        })
    else:
        # 2. Recherche Hybride (Vectoriel + BM25 avec RRF), avec repli élargi automatique
        filtered_chunks, filtered_metas, relaxed_chunks, relaxed_metas, was_relaxed = hybrid_search(
            query=standalone_query,
            collection=collection,
            embedding_model=embedding_model,
            bm25=bm25_index,
            docs=bm25_docs,
            metadatas=bm25_metas,
            chroma_filter=chroma_filter,
            top_k=15
        )

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

        if not source_metadata_list and not relaxed_source_list:
            # Vraiment rien, même en élargissant la recherche : là on peut être honnête,
            # mais on reste dans l'esprit d'ouvrir la discussion plutôt que de la clore.
            no_match_msg = rag_pipeline.build_no_match_message(current_lang)
            with st.chat_message("assistant"):
                st.caption("Mode : réponse IA")
                st.markdown(no_match_msg)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({
                "role": "assistant", "content": no_match_msg, "actual_mode": "generative",
            })
        else:
            if extractive_route:
                extractive_result = None
                extractive_evidence = None
                try:
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
                    full_stream_response = extractive_result.answer_text
                    with st.chat_message("assistant"):
                        st.caption("Réponse extraite des passages sources")
                        st.markdown(full_stream_response)
                        with st.expander(" Ressources consultées"):
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
                    }
                    with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                        audit_f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_stream_response,
                        "sources": display_sources,
                        "answer_mode": "extractive",
                        "actual_mode": "extractive",
                    })
                    st.stop()

            if extractive_route and answer_mode == "Direct answer":
                direct_response = (
                    extractive_result.answer_text
                    if extractive_result is not None
                    else rag_pipeline.build_clarification_message(current_lang)
                )
                with st.chat_message("assistant"):
                    st.caption("Mode : réponse directe")
                    st.markdown(direct_response)
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": direct_response,
                    "actual_mode": "extractive",
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
                        "sources_count": 0,
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
                        ollama,
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

                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_stream_response,
                        "sources": display_sources,
                        "actual_mode": "generative",
                    })

                except Exception as e:
                  st.error(f"Erreur Ollama: {str(e)}. Assurez-vous que Ollama tourne et que '{selected_model}' est installé.")
