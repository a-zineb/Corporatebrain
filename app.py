import streamlit as st
import os

# ==========================================
# 1. CONFIGURATION STREAMLIT (TOUJOURS EN PREMIER)
# ==========================================
st.set_page_config(page_title="Corporate Brain", layout="wide")

import chromadb
import re
import hashlib
import json
import io
import zipfile
from datetime import datetime
import pandas as pd
import docx
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import ollama

# ==========================================
# 2. CONFIGURATION INITIALE & DOSSIERS
# ==========================================
STORAGE_DIR = "doc_storage"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
RELEVANCE_DISTANCE_THRESHOLD = 1.3  # Seuil de distance sémantique

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# ==========================================
# 3. CHARGEMENT DU BACKEND (MiniLM + ChromaDB)
# ==========================================
@st.cache_resource
def load_backend():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, collection, model

client, collection, embedding_model = load_backend()

# ==========================================
# 4. DÉTECTEUR DE LANGUE DYNAMIQUE (FR / EN)
# ==========================================
def detect_query_language(text, fallback_lang="French"):
    """
    Détecte la langue de la question. Si c'est un acronyme seul (ex: 'OCM'), 
    garde la langue du dernier échange pour une transition fluide.
    """
    text_lower = text.lower().strip()
    
    en_keywords = {'what', 'how', 'why', 'who', 'where', 'when', 'is', 'are', 'the', 'this', 'that', 'can', 'you', 'explain', 'tell', 'me', 'show', 'list', 'details', 'about', 'in', 'with', 'for', 'specs', 'specification'}
    fr_keywords = {'que', 'quoi', 'comment', 'pourquoi', 'qui', 'où', 'quand', 'est', 'sont', 'le', 'la', 'les', 'ce', 'cette', 'peux', 'tu', 'expliquer', 'montre', 'donne', 'moi', 'sur', 'dans', 'avec', 'pour', 'spécification', 'spécifications'}
    
    words = set(re.findall(r'\b\w+\b', text_lower))
    en_score = len(words.intersection(en_keywords))
    fr_score = len(words.intersection(fr_keywords))
    
    if en_score > fr_score:
        return "English"
    elif fr_score > en_score:
        return "French"
    else:
        return fallback_lang

# ==========================================
# 5. PARSERS MULTI-FORMATS ET CHUNKING
# ==========================================
def extract_text_from_bytes(file_bytes, file_name):
    ext = os.path.splitext(file_name)[1].lower()
    text_data = []

    try:
        if ext == ".pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                text_data.append((f"Page {page_num + 1}", page.get_text()))
                
        elif ext == ".docx":
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = [para.text for para in doc.paragraphs]
            text_data.append(("Corps du document", "\n".join(full_text)))
            
        elif ext == ".xlsx":
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                text_data.append((f"Feuille: {sheet}", df.to_string()))
                
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
        st.error(f"Erreur de lecture sur {file_name}: {str(e)}")
    
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

def chunk_text_data(parsed_data, chunk_size=800, overlap=150):
    chunks = []
    for location, text in parsed_data:
        if not text.strip():
            continue
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "location": location
            })
            if i + chunk_size >= len(words):
                break
    return chunks

# ==========================================
# 6. INTERFACE SIDEBAR & FILTRES
# ==========================================
if "last_lang" not in st.session_state:
    st.session_state.last_lang = "French"

with st.sidebar:
    st.header(" Configuration & Filtres")
    
    filter_ent = st.selectbox("Zone Géographique (Filiale)", ["Tous", "OCM", "OEG", "OJO", "OCI"])
    filter_application = st.selectbox("Application", ["Tous", "KPSA", "MZ"])
    
    st.markdown("---")
    
    st.subheader("Ingestion de documents")
    uploaded_files = st.file_uploader("PDF, DOCX, XLSX, CSV, ZIP (Chiffré)", accept_multiple_files=True, type=["pdf", "docx", "xlsx", "csv", "zip"])
    
    if uploaded_files:
        for f in uploaded_files:
            file_bytes = f.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            
            # --- CORRECTION BUG DOUBLONS ---
            existing = collection.get(where={"file_hash": file_hash})
            if existing and len(existing["ids"]) > 0:
                st.info(f"{f.name}: Contenu déjà indexé précédemment.")
                continue  # Évite la ré-indexation en doublon
            # -------------------------------
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name, ext = os.path.splitext(f.name)
            saved_filename = f"{base_name}_{timestamp_str}{ext}"
            target_path = os.path.join(STORAGE_DIR, saved_filename)
            
            with open(target_path, "wb") as out_f:
                out_f.write(file_bytes)
            
            parsed_text_data = extract_text_from_bytes(file_bytes, f.name)
            document_chunks = chunk_text_data(parsed_text_data)
            
            if document_chunks:
                ent_tag, app_tag = infer_metadata(f.name)
                ids, embeddings, metadatas, documents = [], [], [], []
                
                for idx, chunk in enumerate(document_chunks):
                    chunk_id = f"{base_name}_{timestamp_str}_chunk_{idx}"
                    ids.append(chunk_id)
                    documents.append(chunk["text"])
                    embeddings.append(embedding_model.encode(chunk["text"]).tolist())
                    metadatas.append({
                        "source_file": f.name,
                        "saved_as": saved_filename,
                        "location": chunk["location"],
                        "geographical_entity": ent_tag,
                        "application": app_tag,
                        "file_hash": file_hash,
                        "timestamp_ingest": timestamp_str
                    })
                
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                st.success(f" Indexation réussie : {f.name}")

    st.markdown("---")
    st.subheader(" Statut du Système")
    st.caption(" Connected: MiniLM + ChromaDB + Qwen 2.5")
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

# ==========================================
# 8. INTERFACE CHAT
# ==========================================
st.title("🧠 Corporate Brain Assistant")
st.markdown("*Base de connaissances locale RAG - Spécifications Atos*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(" Sources & Références / Sources & References"):
                for src in msg["sources"]:
                    st.write(f" **Fichier / File**: {src['file']} ({src['loc']}) | **Score**: {src['score']:.2f}")
                    st.caption(src["text"])

if user_query := st.chat_input("Posez votre question / Ask your question..."):
    
    current_lang = detect_query_language(user_query, fallback_lang=st.session_state.last_lang)
    st.session_state.last_lang = current_lang
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if collection.count() == 0:
        no_docs = "⚠️ Aucun document n'est indexé. / No documents are currently indexed."
        with st.chat_message("assistant"):
            st.warning(no_docs)
        st.session_state.messages.append({"role": "assistant", "content": no_docs})
    else:
        query_vector = embedding_model.encode(user_query).tolist()
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=4,
            where=chroma_filter
        )
        
        valid_context_chunks = []
        source_metadata_list = []
        
        if search_results and search_results["documents"] and len(search_results["documents"][0]) > 0:
            for i in range(len(search_results["documents"][0])):
                distance = search_results["distances"][0][i]
                if distance <= RELEVANCE_DISTANCE_THRESHOLD:
                    valid_context_chunks.append(search_results["documents"][0][i])
                    meta = search_results["metadatas"][0][i] if (search_results.get("metadatas") and search_results["metadatas"][0]) else {}
                    source_metadata_list.append({
                        "file": meta.get("source_file", "Fichier source"),
                        "loc": meta.get("location", "N/A"),
                        "score": 1 / (1 + distance),
                        "text": search_results["documents"][0][i]
                    })
                    
        if not valid_context_chunks:
            prompt_instructions = f"""You are the technical AI assistant 'Corporate Brain'.

STRICT LANGUAGE DIRECTIVE:
- REQUIRED OUTPUT LANGUAGE: {current_lang.upper()}
- YOU MUST RESPOND EXCLUSIVELY IN {current_lang.upper()}. Never output Russian, Cyrillic, or Spanish.

SITUATION:
No matching information was found in the indexed documents for: "{user_query}".

TASK:
Politely explain in {current_lang.upper()} that this information is not available in the documents under current filters."""
        else:
            context_str = "\n\n".join(valid_context_chunks)
            
            prompt_instructions = f"""You are the technical AI assistant 'Corporate Brain'.

STRICT LANGUAGE DIRECTIVE:
- REQUIRED OUTPUT LANGUAGE: {current_lang.upper()}
- YOU MUST RESPOND EXCLUSIVELY IN {current_lang.upper()}.
- IF CONTEXT IS IN FRENCH AND TARGET LANGUAGE IS ENGLISH: Translate and summarize all details accurately into ENGLISH.
- Never output Russian, Cyrillic, or Spanish.

CONTEXT DOCUMENTS:
{context_str}

USER QUESTION / ACRONYM:
{user_query}"""

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_stream_response = ""
            
            try:
                # --- OPTIMISATION : PASSE SUR QWEN2.5:7B ---
                llm_stream = ollama.chat(
                    model="qwen2.5:7b",
                    messages=[{"role": "user", "content": prompt_instructions}],
                    options={"temperature": 0.0},
                    stream=True
                )
                
                for chunk in llm_stream:
                    full_stream_response += chunk["message"]["content"]
                    response_placeholder.markdown(full_stream_response + "▌")
                
                response_placeholder.markdown(full_stream_response)
                
                if source_metadata_list:
                    with st.expander(" Sources & Références / Sources & References"):
                        for src in source_metadata_list:
                            st.write(f" **Fichier / File**: {src['file']} ({src['loc']}) | **Score**: {src['score']:.2f}")
                            st.caption(src["text"])
                        
                audit_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "question": user_query,
                    "language": current_lang,
                    "sources_count": len(source_metadata_list)
                }
                with open("audit_log.jsonl", "a", encoding="utf-8") as audit_f:
                    audit_f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_stream_response,
                    "sources": source_metadata_list
                })
                
            except Exception as e:
                st.error(f"Ollama Error: {str(e)}")