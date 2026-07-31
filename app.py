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

import rag_pipeline

# Fonction pour ouvrir un fichier local
def open_local_file(path):
    try:
        if os.path.exists(path):
            os.startfile(path)
    except Exception as e:
        print(f"Erreur d'ouverture de fichier : {e}")

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


def list_catalog_documents(collection, chroma_filter=None):
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
    return sorted(unique.values(), key=lambda item: (item.get("application", ""), item.get("geographical_entity", ""), item.get("source_file", "").lower()))


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

if user_query := st.chat_input("Posez votre question ou tapez un acronyme..."):

    current_lang = detect_query_language(user_query, fallback_lang=st.session_state.last_lang)
    st.session_state.last_lang = current_lang

    with st.chat_message("user"):
        st.markdown(user_query)

    # 1. Reformulation de la question
    standalone_query = contextualize_query(user_query, st.session_state.messages, selected_model)

    if collection.count() == 0:
        no_docs = " Aucun document n'est indexé dans 'doc_storage_v2'." if current_lang == "French" else "⚠️ No documents indexed in 'doc_storage_v2'."
        with st.chat_message("assistant"):
            st.warning(no_docs)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": no_docs})
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
                st.markdown(no_match_msg)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": no_match_msg})
        else:
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
                    }
                    with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                        audit_f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")

                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_stream_response,
                        "sources": display_sources
                    })

                except Exception as e:
                  st.error(f"Erreur Ollama: {str(e)}. Assurez-vous que Ollama tourne et que '{selected_model}' est installé.")
