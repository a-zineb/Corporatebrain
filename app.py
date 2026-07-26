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
from rank_bm25 import BM25Okapi
import docx2txt
import olefile
import struct

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
    all_data = collection.get(include=["documents", "metadatas"])

    # Si la base est vide
    if not all_data["documents"]:
        return None, None, None

    docs = all_data["documents"]

    # BM25 a besoin de listes de mots (tokens) en minuscules
    tokenized_corpus = [doc.lower().split() for doc in docs]

    # On initialise l'algorithme
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, docs, all_data["metadatas"]


def hybrid_search(query, collection, embedding_model, bm25, docs, metadatas, chroma_filter=None, top_k=5,
                   min_results_before_relax=3):
    """
    Recherche hybride (vectorielle + BM25 avec fusion RRF).
    NOUVEAU : si la recherche filtrée renvoie trop peu de résultats, on relance
    automatiquement une passe "élargie" (sans filtre de Zone/Application) pour
    proposer des pistes proches plutôt que de rentrer bredouille.
    Retourne : (docs_filtrés, metas_filtrés, docs_elargis, metas_elargis, relaxed)
    """
    def _run(query, chroma_filter, top_k):
        rrf_scores = {}
        doc_to_meta = {}

        # ----------------------------------------------------
        # 1. RECHERCHE VECTORIELLE (Le "Sens")
        # ----------------------------------------------------
        query_vector = embedding_model.encode(query).tolist()
        vec_results = collection.query(query_embeddings=[query_vector], n_results=10, where=chroma_filter)

        if vec_results and vec_results["documents"] and len(vec_results["documents"][0]) > 0:
            for rank, (doc_text, meta) in enumerate(zip(vec_results["documents"][0], vec_results["metadatas"][0])):
                rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1 / (rank + 1 + 60))
                doc_to_meta[doc_text] = meta

        # ----------------------------------------------------
        # 2. RECHERCHE BM25 (Le mot-clé exact)
        # ----------------------------------------------------
        if bm25 is not None and docs is not None:
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

            bm25_count = 0
            for idx in sorted_indices:
                if bm25_count >= 10:
                    break
                if bm25_scores[idx] <= 0:
                    break

                meta = metadatas[idx]

                if chroma_filter:
                    match = True
                    if "$and" in chroma_filter:
                        for condition in chroma_filter["$and"]:
                            for k, v in condition.items():
                                if meta.get(k) != v:
                                    match = False
                                    break
                    else:
                        for k, v in chroma_filter.items():
                            if meta.get(k) != v:
                                match = False
                                break
                    if not match:
                        continue

                doc_text = docs[idx]
                rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1 / (bm25_count + 1 + 60))
                doc_to_meta[doc_text] = meta
                bm25_count += 1

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_docs, final_metas = [], []
        for doc_text, score in sorted_docs[:top_k]:
            final_docs.append(doc_text)
            final_metas.append(doc_to_meta[doc_text])

        return final_docs, final_metas

    # --- Passe principale (avec filtres actifs) ---
    filtered_docs, filtered_metas = _run(query, chroma_filter, top_k)

    # --- Passe élargie : uniquement si la passe filtrée est pauvre, et seulement
    # si un filtre était réellement actif (sinon ça reviendrait au même résultat) ---
    relaxed_docs, relaxed_metas = [], []
    relaxed = False
    if chroma_filter is not None and len(filtered_docs) < min_results_before_relax:
        relaxed = True
        relaxed_docs, relaxed_metas = _run(query, None, top_k)
        # On retire les doublons déjà présents dans les résultats filtrés
        seen = set(filtered_docs)
        dedup_docs, dedup_metas = [], []
        for d, m in zip(relaxed_docs, relaxed_metas):
            if d not in seen:
                dedup_docs.append(d)
                dedup_metas.append(m)
        relaxed_docs, relaxed_metas = dedup_docs, dedup_metas

    return filtered_docs, filtered_metas, relaxed_docs, relaxed_metas, relaxed


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
    """Reformule la question utilisateur pour la rendre autonome grâce à l'historique."""
    if not chat_history:
        return user_query

    recent_history = ""
    for msg in chat_history[-3:]:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        recent_history += f"{role}: {msg['content']}\n"

    prompt_rewrite = f"""Compte tenu de l'historique de conversation suivant et de la dernière question de l'utilisateur, reformule la dernière question pour qu'elle soit totalement AUTONOME et COMPRÉHENSIBLE sans l'historique (remplace les pronoms comme 'it', 'ce terme', 'celui-ci', 'the answer' par les acronymes ou sujets réels abordés précédemment).
Si la question est déjà autonome, renvoie-la exactement à l'identique.
Ne réponds pas à la question, renvoie UNIQUEMENT la question reformulée.

HISTORIQUE :
{recent_history}

QUESTION : {user_query}
QUESTION REFORMULÉE :"""

    try:
        res = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_rewrite}],
            options={"temperature": 0.0}
        )
        reformulated = res["message"]["content"].strip()
        return reformulated if reformulated else user_query
    except Exception as e:
        print(f"Ollama Error in contextualize_query: {e}")
        return user_query

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

        def build_source_list(chunks, metas, relaxed_flag=False, start_id=1):
            out = []
            for i, (doc_text, meta) in enumerate(zip(chunks, metas)):
                filename = meta.get("source_file", "Fichier source")
                out.append({
                    "id": start_id + i,
                    "file": filename,
                    "loc": meta.get("location", "N/A"),
                    "text": doc_text,
                    "path": os.path.abspath(os.path.join(STORAGE_DIR, filename)),
                    "relaxed": relaxed_flag
                })
            return out

        source_metadata_list = build_source_list(filtered_chunks, filtered_metas, relaxed_flag=False)
        relaxed_source_list = build_source_list(
            relaxed_chunks, relaxed_metas, relaxed_flag=True, start_id=len(source_metadata_list) + 1
        )
        all_sources_for_prompt = source_metadata_list + relaxed_source_list

        if not source_metadata_list and not relaxed_source_list:
            # Vraiment rien, même en élargissant la recherche : là on peut être honnête,
            # mais on reste dans l'esprit d'ouvrir la discussion plutôt que de la clore.
            no_match_msg = (
                "I couldn't find anything close to that in the indexed documents, even outside the current "
                "filters. Could you rephrase your question, try a related keyword, or tell me the department/"
                "topic you're aiming for? That would help me point you in the right direction."
                if current_lang == "English"
                else "Je n'ai rien trouvé de proche dans les documents indexés, même en élargissant la recherche "
                     "au-delà des filtres actifs. Peux-tu reformuler ta question, essayer un mot-clé associé, ou "
                     "me préciser le service/sujet visé ? Ça m'aiderait à t'orienter."
            )
            with st.chat_message("assistant"):
                st.markdown(no_match_msg)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": no_match_msg})
        else:
            context_chunks_formatted = []
            for src in all_sources_for_prompt:
                tag = " (hors des filtres actifs — piste proche)" if src.get("relaxed") else ""
                context_chunks_formatted.append(f"[SOURCE {src['id']}]{tag}\n{src['text']}")
            context_str = "\n---\n".join(context_chunks_formatted)

            recent_chat_history = ""
            for m in st.session_state.messages[-4:]:
                role = "Utilisateur" if m["role"] == "user" else "Assistant"
                recent_chat_history += f"{role}: {m['content']}\n"

            relaxed_note = (
                "\nNOTE IMPORTANTE : certaines sources ci-dessus sont marquées '(hors des filtres actifs — piste "
                "proche)'. Elles ne correspondent pas exactement aux filtres Zone/Application sélectionnés, mais "
                "peuvent constituer une piste utile à proposer à l'utilisateur (mentionne-le clairement, par "
                "exemple : \"je n'ai rien trouvé pile dans [filtre], mais j'ai trouvé quelque chose de proche dans "
                "[autre catégorie], est-ce que ça pourrait t'intéresser ?\")."
                if was_relaxed else ""
            )

            prompt_instructions = f"""Tu es l'assistant technique d'entreprise 'Corporate Brain'. Ton ton est celui d'un collègue serviable qui engage la discussion, pas celui d'un moteur de recherche binaire qui répond juste "trouvé" ou "pas trouvé".

PROJET & FILTRES ACTIFS :
- Zone Géographique (Filiale) : {filter_ent}
- Application : {filter_application}

HISTORIQUE RÉCENT DE LA CONVERSATION :
{recent_chat_history}

CONTEXTE DOCUMENTAIRE :
{context_str}
{relaxed_note}

DERNIÈRE QUESTION DE L'UTILISATEUR :
{user_query}

INSTRUCTIONS :
1. Si l'information exacte demandée est présente dans le CONTEXTE, réponds directement et cite la ou les sources au format [SOURCE X].
2. Si l'information exacte n'est pas présente mais que tu repères des éléments proches, apparentés, ou des synonymes/catégories voisines dans le CONTEXTE (par exemple : l'utilisateur cherche "département IT" et tu trouves des mentions de "Technologie", "Informatique", "Systèmes d'Information", ou un acronyme lié), NE REFUSE PAS SÈCHEMENT. Propose plutôt ces pistes de façon naturelle et conversationnelle, en citant leur source [SOURCE X], et explique en quoi elles pourraient correspondre à la demande.
3. Tu as le droit de faire des rapprochements logiques entre plusieurs fragments du CONTEXTE pour construire ta réponse ou tes suggestions.
4. Termine par une question ouverte ou une invitation à préciser si tu n'es pas sûr à 100% (ex : "Est-ce que ça correspond à ce que tu cherches ?" ou "Veux-tu que je regarde plus précisément du côté de X ?"), afin d'entretenir la discussion plutôt que de la clore abruptement.
5. Utilise uniquement les informations du CONTEXTE DOCUMENTAIRE ci-dessus (pas de connaissances externes/génériques), mais reste ouvert et exploratoire avec ce qui s'y trouve plutôt que strictement littéral.
6. Ne dis "je n'ai rien trouvé" que si le CONTEXTE ne contient vraiment rien qui se rapporche même de loin au sujet de la question — et dans ce cas, propose quand même une piste (reformulation, mot-clé à essayer, ou filtre à changer) plutôt que de t'arrêter là.
7. Ne fais aucune remarque sur ton identité d'IA ou tes limites de date.
8. Réponds dans la même langue que la question de l'utilisateur ({current_lang}).
"""
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_stream_response = ""

                try:
                    llm_stream = ollama.chat(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt_instructions}],
                        options={"temperature": 0.2},
                        stream=True
                    )

                    for chunk in llm_stream:
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            full_stream_response += content
                            response_placeholder.markdown(full_stream_response + "▌")
                    response_placeholder.markdown(full_stream_response)

                    # Extraire les citations [SOURCE X] de la réponse de l'IA
                    cited_ids = [int(num) for num in re.findall(r'\[SOURCE (\d+)\]', full_stream_response)]
                    cited_ids = list(set(cited_ids))

                    # Une réponse qui indique explicitement que l'information n'est pas
                    # couverte par les documents ne doit pas afficher de ressources,
                    # même si le modèle a ajouté des balises de citation.
                    response_lower = full_stream_response.lower()
                    no_coverage_patterns = [
                        r"je ne trouve pas.*(document|contexte|source|corpus)",
                        r"n.est pas.*(document|contexte|source|corpus)",
                        r"information.*(non couverte|absente|indisponible)",
                        r"le contexte( fourni)? ne contient aucune information",
                        r"le contexte fourni ne contient pas",
                        r"les documents ne contiennent aucune information",
                        r"ne trouve pas de r.ponse dans le contexte( fourni)?",
                        r"la question pos.e ne trouve pas de r.ponse dans le contexte( fourni)?",
                        r"ne trouve pas de r.ponse dans les documents",
                        r"n.est pas mentionn. dans les documents",
                        r"n.est pas abord. dans les documents",
                        r"l.information n.est pas pr.sente dans les documents",
                        r"(i cannot|i can.t|not found|not covered).*(document|context|source|corpus)",
                        r"(not mentioned|not available).*(document|context|source|corpus)",
                    ]
                    no_documentary_answer = any(
                        re.search(pattern, response_lower, flags=re.DOTALL)
                        for pattern in no_coverage_patterns
                    )

                    # Si l'IA a cité des sources, on filtre.
                    # Sinon, aucune source n'est affichée.
                    display_sources = []

                    if no_documentary_answer:
                        display_sources = []
                    elif cited_ids:
                        display_sources = [
                            src for src in all_sources_for_prompt
                            if src["id"] in cited_ids
                        ]

                    # DEBUG TEMPORAIRE
                    print("\n========== SOURCES DEBUG ==========")
                    print("ALL SOURCES:", len(all_sources_for_prompt))
                    print("CITED IDS:", cited_ids)
                    print("DISPLAY SOURCES:", len(display_sources))

                    print("\n--- ALL SOURCES ---")
                    for src in all_sources_for_prompt:
                        print(
                            "ID:", src["id"],
                            "| File:", src["file"],
                            "| Location:", src["loc"],
                            "| Text:", src["text"][:200],
                            "| Path:", src["path"],
                            "| Relaxed:", src["relaxed"]
                        )

                    print("\n--- DISPLAY SOURCES ---")
                    for src in display_sources:
                        print(
                            "ID:", src["id"],
                            "| File:", src["file"],
                            "| Location:", src["loc"],
                            "| Text:", src["text"][:200],
                            "| Path:", src["path"],
                            "| Relaxed:", src["relaxed"]
                        )

                    print("===================================\n")
                    unique_sources = {}
                    for src in display_sources:
                        if src["path"] not in unique_sources:
                            unique_sources[src["path"]] = src

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
