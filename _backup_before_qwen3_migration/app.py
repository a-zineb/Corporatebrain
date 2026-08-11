# ============================================================
# CORPORATE BRAIN — app.py
# SaaS RAG pour Atos | Architecture 3 colonnes V1
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
from app import quick_prompts
import streamlit as st
import chromadb
import fitz  # PyMuPDF
import ollama
from sentence_transformers import SentenceTransformer

# ── 2. CONFIGURATION GLOBALE DE LA PAGE ─────────────────────
st.set_page_config(
    page_title="Corporate Brain | Atos",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 3. INITIALISATION BACKEND (mis en cache) ─────────────────
@st.cache_resource
def load_backend():
    # Chargement explicite du modèle d'embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("documents")
    return model, collection

model, collection = load_backend()

# ── 3bis. FONCTION DE CHUNKING ───────────────────────────────
def chunk_text(text, size=300, overlap=50):
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + size])
    return chunks

# ── 4. SESSION STATE (mémoire de chat en RAM) ────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ══════════════════════════════════════════════════════════════
#  COLONNE GAUCHE — SIDEBAR (Zone Admin)
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("###  Corporate Brain")
    st.caption("Powered by Atos × Ollama")
    st.divider()

    st.subheader(" Document Management")
    uploaded_files = st.file_uploader(
        "Ajouter des documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Formats supportés : PDF"
    )

    # ── TRAITEMENT RÉEL DES FICHIERS ─────────────────────────
    if uploaded_files:
        for file in uploaded_files:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()

            chunks = chunk_text(text)

            if chunks:
                # Génération des vrais vecteurs d'embeddings pour ChromaDB
                embeddings = model.encode(chunks).tolist()
                ids = [f"{file.name}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"source": file.name} for _ in chunks]
                
                collection.add(
                    embeddings=embeddings, 
                    documents=chunks, 
                    ids=ids, 
                    metadatas=metadatas
                )

        st.success(f"{len(uploaded_files)} document(s) indexé(s) ✅")

    st.divider()

    st.subheader(" Base de connaissances")
    doc_count = collection.count()
    if doc_count > 0:
        st.success(f" {doc_count} chunks indexés")
    else:
        st.warning(" Base vide — uploadez un document")

# ══════════════════════════════════════════════════════════════
#  ZONE PRINCIPALE — 2 colonnes : chat (75%) + insights (25%)
# ══════════════════════════════════════════════════════════════
col_chat, col_insights = st.columns([3, 1])

with col_chat:
    st.markdown("##  Assistant Corporate Brain")
    st.caption("Posez vos questions sur les documents Atos indexés.")
    st.divider()

    # Affichage de l'historique du chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("🔍 Sources utilisées"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** : _{src}_")

    # Entrée utilisateur
    if prompt := st.chat_input("Posez votre question sur les documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):

                # 1. Vectorisation de la question de l'utilisateur
                query_embedding = model.encode([prompt]).tolist()

                # 2. VRAI RETRIEVAL basé sur l'embedding généré
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Extraction sécurisée des données de ChromaDB
                found_chunks = results["documents"][0] if results["documents"] else []
                found_metadatas = results["metadatas"][0] if results["metadatas"] else []
                found_distances = results["distances"][0] if results["distances"] else []

                # ── SEUIL DE PERTINENCE VIA VECTOR DISTANCE ──────
                SEUIL_DISTANCE = 1.1

                if not found_chunks or len(found_distances) == 0 or found_distances[0] > SEUIL_DISTANCE:
                    response = "Je ne trouve pas cette information dans les documents indexés."
                    sources = []
                else:
                    sources = list(set([meta["source"] for meta in found_metadatas]))
                    context = "\n\n".join(found_chunks)

                    # Prompt strict renforcé
                    rag_prompt = f"""Tu es l'assistant IA "Corporate Brain" spécialisé dans l'analyse de documents d'entreprise.
Instructions strictes :
1. Réponds à la question posée UNIQUEMENT en te basant sur le contexte fourni ci-dessous.
2. Si la réponse n'est pas directement et explicitement mentionnée dans le contexte, tu dois répondre EXACTEMENT ceci : "Je ne trouve pas cette information dans les documents indexés."
3. Ne fais aucune supposition, n'extrapole pas et n'utilise JAMAIS tes connaissances générales.

Contexte :
{context}

Question : {prompt}

Réponse :"""

                    # Appel unique et définitif à Ollama avec température 0
                    llm_response = ollama.chat(
                        model="llama3", 
                        messages=[
                            {"role": "user", "content": rag_prompt}
                        ],
                        options={
                            "temperature": 0.0
                        }
                    )
                    response = llm_response["message"]["content"]
                # ── AFFICHAGE EN TEMPS RÉEL ─────────────────────
                st.markdown(response)
                if sources:
                    with st.expander("🔍 Sources utilisées"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** : _{src}_")

        # Sauvegarde dans la mémoire de la session
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

with col_insights:
    st.markdown("###  Statut Système")
    st.metric(label="LLM", value="Ollama", delta="Llama 3 · Local")
    st.metric(label="Embeddings", value="MiniLM-L6", delta="Sentence Transformers")
    st.divider()

   # st.markdown("### Suggestions")
    #quick_prompts = [
        #"Quels sont les objectifs du projet ?",
        #"Résume les points clés de ce document",
       # "Quels sont les risques identifiés ?",
        #"Qui sont les parties prenantes ?",
    #]
    for qp in quick_prompts:
        if st.button(qp, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": qp})
            st.rerun()