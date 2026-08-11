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
import openai
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
st.set_page_config(page_title="Corporate Brain V2 - Ultra Rapide", layout="wide")

api_key = st.sidebar.text_input("🔑 Clé API OpenAI (sk-...)", type="password")
if not api_key:
    st.warning("⚠️ Veuillez entrer une clé API OpenAI dans la barre latérale pour activer Corporate Brain.")
    st.stop()

class OpenAIEmbeddingWrapper:
    def __init__(self, key):
        self.client = openai.OpenAI(api_key=key)
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        res = self.client.embeddings.create(input=texts, model="text-embedding-3-small")
        embs = [d.embedding for d in res.data]
        return embs[0] if len(texts) == 1 else embs

# ==========================================
# 2. CONFIGURATION INITIALE & DOSSIERS
# ==========================================
STORAGE_DIR = "doc_storage_v2"
CHROMA_PATH = "chroma_db_openai"
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
def load_backend(key):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    
    embedding_model = OpenAIEmbeddingWrapper(key)
    
    return client, collection, embedding_model

try:
    with st.spinner(" Chargement du moteur de recherche ultra-rapide..."):
        client, collection, embedding_model = load_backend(api_key)
    # Test rapide pour vérifier que la collection est valide
    collection.count()
except Exception:
    # La collection a été réinitialisée manuellement : on vide le cache et on recharge
    load_backend.clear()
    with st.spinner(" Réinitialisation de la base de données..."):
        client, collection, embedding_model = load_backend(api_key)

def build_bm25_index(collection):
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
def hybrid_search(query, collection, embedding_model, bm25, docs, metadatas, chroma_filter=None, top_k=5):
    # Dictionnaire pour stocker le score final de chaque texte
    rrf_scores = {}
    doc_to_meta = {} # Pour garder une trace de la source de chaque texte
    
    # ----------------------------------------------------
    # 1. RECHERCHE VECTORIELLE (Le "Sens")
    # ----------------------------------------------------
    query_vector = embedding_model.encode(query)
    vec_results = collection.query(query_embeddings=[query_vector], n_results=10, where=chroma_filter)
    
    if vec_results and vec_results["documents"] and len(vec_results["documents"][0]) > 0:
        for rank, (doc_text, meta) in enumerate(zip(vec_results["documents"][0], vec_results["metadatas"][0])):
            # Formule RRF : 1 / (rang + 60)
            rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1 / (rank + 1 + 60))
            doc_to_meta[doc_text] = meta

    # ----------------------------------------------------
    # 2. RECHERCHE BM25 (Le mot-clé exact)
    # ----------------------------------------------------
    if bm25 is not None and docs is not None:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        # Trier TOUS les documents par score BM25 décroissant
        sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        bm25_count = 0
        for idx in sorted_indices:
            if bm25_count >= 10:
                break
                
            if bm25_scores[idx] <= 0:
                break # Les scores suivants seront aussi <= 0
                
            meta = metadatas[idx]
            
            # Application manuelle du chroma_filter pour garantir l'étanchéité
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

    # ----------------------------------------------------
    # 3. FUSION ET TRI
    # ----------------------------------------------------
    # On trie les documents du plus grand score RRF au plus petit
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # On garde seulement les "top_k" (les 5 meilleurs par défaut)
    final_docs = []
    final_metas = []
    for doc_text, score in sorted_docs[:top_k]:
        final_docs.append(doc_text)
        final_metas.append(doc_to_meta[doc_text])
        
    return final_docs, final_metas

# ==========================================
# 4. FONCTIONS UTILITAIRES (DÉTECTION, MODELS, REFORMULATION)
# ==========================================
# Modèles locaux supprimés au profit de OpenAI

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
        openai_client = openai.OpenAI(api_key=api_key)
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_rewrite}],
            temperature=0.0
        )
        reformulated = res.choices[0].message.content.strip()
        return reformulated if reformulated else user_query
    except Exception:
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
    
    # Extraire toutes les séquences de caractères ANSI/latin-1 lisibles (>= 4 chars)
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
                # Faux .docx (vieux format .doc déguisé) → on tente olefile
                try:
                    extracted = _extract_ole_doc(file_bytes)
                    if extracted.strip():
                        text_data.append(("Corps du document", extracted))
                except Exception:
                    pass  # Fichier non lisible, ignoré silencieusement

        elif ext == ".doc":
            # Vieux format binaire .doc → extraction via olefile
            try:
                extracted = _extract_ole_doc(file_bytes)
                if extracted.strip():
                    text_data.append(("Corps du document", extracted))
            except Exception:
                pass  # Fichier non lisible, ignoré silencieusement
            
        elif ext == ".xlsx":
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet).fillna("")
                # Convertit chaque ligne en une phrase "Colonne: Valeur, Colonne: Valeur."
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
                # 1. On sépare d'abord par paragraphes
        paragraphs = text.split('\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 2. Si le paragraphe est immense, on le coupe en phrases
            if len(para) > max_length:
                sentences = split_into_sentences(para)
            else:
                # Sinon, on le traite comme un bloc entier
                sentences = [para]
                
            # 3. On recombine les phrases intelligemment
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Si ajouter cette phrase dépasse la limite, on sauvegarde le chunk actuel
                if len(current_chunk) + len(sentence) > max_length and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "location": location
                    })
                    # OVERLAP: on récupère les ~250 derniers caractères pour garder le contexte
                    overlap = current_chunk[-250:] if len(current_chunk) > 250 else current_chunk
                    if " " in overlap:
                        overlap = overlap[overlap.find(" ")+1:] # Éviter de couper un mot au milieu
                    
                    current_chunk = overlap.strip() + " " + sentence + " "
                else:
                    # Sinon, il y a de la place, on l'ajoute au chunk en cours
                    current_chunk += sentence + " "
                    
        # 4. À la toute fin du document, on n'oublie pas de sauvegarder le dernier morceau
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
                    
                    # ENRICHISSEMENT SÉMANTIQUE : On injecte le contexte métier dans le texte lui-même
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
                
                # BATCHING INTELLIGENT : Pour éviter la limite de Tokens Par Minute (TPM) d'OpenAI
                import time
                import openai
                
                embeddings = []
                batch_size = 10 # 10 chunks par requête = environ 3000 tokens
                
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    
                    # Boucle de sécurité (retry) en cas d'erreur API temporaire
                    while True:
                        try:
                            batch_embs = embedding_model.encode(batch_docs)
                            if not isinstance(batch_embs[0], list):
                                batch_embs = [batch_embs]
                            embeddings.extend(batch_embs)
                            
                            # Petite pause volontaire pour ne pas saturer l'API OpenAI (Tokens Per Minute limit)
                            time.sleep(1.5)
                            break
                        except openai.RateLimitError:
                            print("⚠️ Limite OpenAI atteinte. L'application se met en pause 20 secondes...")
                            time.sleep(20)
                        except Exception as e:
                            print(f"Erreur API inattendue : {e}")
                            break
                
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

sync_local_folder_v2()
bm25_index, bm25_docs, bm25_metas = build_bm25_index(collection)

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
    st.info("🧠 GPT-4o-mini (OpenAI)")
    selected_model = "gpt-4o-mini"
    
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
st.markdown("*RAG Optimisé : Réponses Instantanées & Extrait Direct*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"] and "L'information recherchée n'est pas présente" not in msg["content"]:
            with st.expander(" Ressources consultées"):
                # Dédoublonnage par fichier pour n'afficher que les ressources uniques
                unique_sources = {}
                for src in msg["sources"]:
                    if src["path"] not in unique_sources:
                        unique_sources[src["path"]] = src
                
                for i_src, src in enumerate(unique_sources.values()):
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.write(f" **Fichier**: {src['file']}")
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
                # 2. Recherche Hybride (Vectoriel + BM25 avec RRF)
        valid_context_chunks, raw_metas = hybrid_search(
            query=standalone_query,
            collection=collection,
            embedding_model=embedding_model,
            bm25=bm25_index,
            docs=bm25_docs,
            metadatas=bm25_metas,
            chroma_filter=chroma_filter,
            top_k=15
        )
        
        source_metadata_list = []
        for i, (doc_text, meta) in enumerate(zip(valid_context_chunks, raw_metas)):
            filename = meta.get("source_file", "Fichier source")
            source_metadata_list.append({
                "id": i + 1,
                "file": filename,
                "loc": meta.get("location", "N/A"),
                "text": doc_text,
                "path": os.path.abspath(os.path.join(STORAGE_DIR, filename))
            })

                    
        refusal_msg = (
            "The requested information is not explicitly present in the indexed documents."
            if current_lang == "English"
            else "L'information recherchée n'est pas présente de façon explicite dans les documents indexés."
        )

        if not valid_context_chunks:
            full_stream_response = refusal_msg
            with st.chat_message("assistant"):
                st.markdown(full_stream_response)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": full_stream_response})
        else:
            context_chunks_formatted = []
            for src in source_metadata_list:
                context_chunks_formatted.append(f"[SOURCE {src['id']}]\n{src['text']}")
            context_str = "\n---\n".join(context_chunks_formatted)
            
            # Reconstruction de l'historique récent pour le prompt final
            recent_chat_history = ""
            for m in st.session_state.messages[-4:]:
                role = "Utilisateur" if m["role"] == "user" else "Assistant"
                recent_chat_history += f"{role}: {m['content']}\n"
            
            prompt_instructions = f"""Tu es l'assistant technique d'entreprise 'Corporate Brain'.

PROJET & FILTRES ACTIFS :
- Zone Géographique (Filiale) : {filter_ent}
- Application : {filter_application}

RÈGLES STRICTES DE RÉPONSE :
1. Réponds EXCLUSIVEMENT en {current_lang.upper()}.
2. Sois CONCIS, EXCLUSIVEMENT FACTUEL et DIRECT.
3. Basse ta réponse SUR LE CONTEXTE DOCUMENTAIRE fourni et l'HISTORIQUE RÉCENT.
4. Les documents fournis ont été filtrés pour la Zone '{filter_ent}' et l'Application '{filter_application}'. Ne réponds qu'aux éléments directement liés à ces filtres. Si l'information demandée concerne une autre zone ou application, refuse de répondre.
5. SI L'INFORMATION N'EST PAS DANS LE CONTEXTE DOCUMENTAIRE : Réponds UNIQUEMENT : "{refusal_msg}". Tu as l'interdiction absolue de parler de ta date d'entraînement, de tes limites de connaissances, ou de ton créateur (ex: Microsoft, OpenAI). Tu dois simplement dire que l'information n'est pas dans les documents.
6. SI UN TERME OU ACRONYME (ex: KAABU, OCM) EST MENTIONNÉ DANS LES DOCUMENTS, MAIS SANS DÉFINITION DITCTIONNAIRE EXPLICITE : Ne dis PAS que l'information est absente. Résume précisément son utilisation, ses cas de test, ses modules ou le contexte technique dans lequel il apparaît.
7. INTERDICTION D'INVENTER : Ne devine jamais la signification développée d'un acronyme via tes connaissances générales externes si elle n'est pas écrite noir sur blanc dans le contexte.
8. Si l'utilisateur demande la traduction ou l'explication d'un terme déjà mentionné précédemment, réponds-lui directement.
9. Si et seulement si le sujet demandé n'a STRICTEMENT AUCUN RAPPORT ou n'est PAS MENTIONNÉ du tout dans le CONTEXTE DOCUMENTAIRE, réponds EXACTEMENT : "{refusal_msg}"
10. INTELLIGENCE CONVERSATIONNELLE : Si l'utilisateur demande "plus de ressources", "d'autres sources" ou "quelles sont les ressources utilisées", NE FAIS PAS juste une liste bête. Analyse sa demande grâce à l'historique : s'il veut plus de détails sur le sujet précédent, fournis-lui de nouvelles informations tirées du contexte. S'il veut savoir d'où tu tires tes infos, explique-lui intelligemment sur quels documents précis (parmi le contexte) tu t'es basé, de manière naturelle et argumentée.
11. ATTENTION : Le contexte peut contenir des données brutes (ex: 'Colonne: Valeur'). Tu dois faire l'effort de lire et déduire la réponse à partir de ces paires clés-valeurs.

HISTORIQUE RÉCENT DE LA CONVERSATION :
{recent_chat_history}

CONTEXTE DOCUMENTAIRE :
{context_str}

DERNIÈRE QUESTION DE L'UTILISATEUR :
{user_query}

RAPPEL : Tu as le droit (et le devoir) de faire des liens logiques entre différents fragments du CONTEXTE pour déduire la réponse.
OBLIGATION ABSOLUE : Dès que tu utilises une information provenant d'un fragment du contexte, tu DOIS citer la source à la fin de ta phrase sous le format exact [SOURCE X] (où X est le numéro de la source).
Si l'information n'est pas présente dans le CONTEXTE DOCUMENTAIRE, réponds : "{refusal_msg}". N'utilise AUCUNE connaissance externe, et ne fais aucune phrase sur ton identité d'IA ou tes limites de date.
"""
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_stream_response = ""
                
                try:
                    openai_client = openai.OpenAI(api_key=api_key)
                    llm_stream = openai_client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt_instructions}],
                        temperature=0.0,
                        stream=True
                    )
                    
                    for chunk in llm_stream:
                        if chunk.choices[0].delta.content:
                            full_stream_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_stream_response + "▌")
                    response_placeholder.markdown(full_stream_response)
                        
                    # Extraire les citations [SOURCE X] de la réponse de l'IA
                    cited_ids = [int(num) for num in re.findall(r'\[SOURCE (\d+)\]', full_stream_response)]
                    cited_ids = list(set(cited_ids)) # Dédoublonnage
                    
                    # Si l'IA a cité des sources, on filtre. Sinon on garde tout.
                    if cited_ids:
                        source_metadata_list = [src for src in source_metadata_list if src["id"] in cited_ids]
                    
                    # Dédoublonnage par fichier pour n'afficher que les ressources uniques
                    unique_sources = {}
                    for src in source_metadata_list:
                        if src["path"] not in unique_sources:
                            unique_sources[src["path"]] = src
                    
                    # On affiche l'expander uniquement si ce n'est pas le message de refus
                    if "L'information recherchée n'est pas présente" not in full_stream_response:
                        with st.expander(" Ressources consultées"):
                            for i_src, src in enumerate(unique_sources.values()):
                                cols = st.columns([3, 1, 1])
                                with cols[0]:
                                    st.write(f" **Fichier**: {src['file']}")
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
                        "sources_count": len(source_metadata_list)
                    }
                    with open("audit_log_v2.jsonl", "a", encoding="utf-8") as audit_f:
                        audit_f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                        
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_stream_response,
                        "sources": source_metadata_list
                    })
                    
                except Exception as e:
                    st.error(f"Erreur Ollama: {str(e)}. Assurez-vous que le modèle '{selected_model}' est bien actif.")