# 🧠 Corporate Brain — Assistant RAG d'Entreprise

> Application RAG (Retrieval-Augmented Generation) développée dans le cadre d'un PFA à **Atos**.
> Permet à un employé de poser des questions en langage naturel sur les documents internes de l'entreprise et d'obtenir une réponse générée par IA, avec les sources citées.

---

## ✨ Fonctionnalités

- 📄 **Upload de documents** : PDF, DOCX, DOC (Word 97), XLSX, CSV, ZIP
- 🔍 **Recherche hybride** : Vectorielle (Sentence Transformers) + BM25 (mots-clés) avec fusion RRF
- 🤖 **Génération de réponses** : via LLM local Ollama (Qwen 2.5, Llama 3, Mistral...)
- 🗂️ **Filtres** : par Zone Géographique (OCM, OEG, OJO, OCI) et par Application (MZ, KPSA)
- 📎 **Sources citées** : chaque réponse affiche les fichiers sources avec bouton d'ouverture directe
- 🛡️ **Anti-hallucination** : l'IA refuse de répondre si l'information n'est pas dans les documents
- 💬 **Mémoire conversationnelle** : reformulation des questions avec l'historique du chat

---

## 🛠️ Stack Technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Extraction PDF | PyMuPDF (fitz) |
| Extraction DOCX | python-docx |
| Extraction DOC (Word 97) | olefile |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Base Vectorielle | ChromaDB |
| Recherche Mots-Clés | BM25 (rank_bm25) |
| LLM | Ollama (local) |

---

## 🚀 Installation

### 1. Prérequis
- Python 3.10+
- [Ollama](https://ollama.ai) installé et un modèle téléchargé (`qwen2.5:7b` recommandé)

```bash
ollama pull qwen2.5:7b
```

### 2. Cloner le projet

```bash
git clone https://github.com/<TON_USERNAME>/Corporatebrain.git
cd Corporatebrain
```

### 3. Créer l'environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# ou
source venv/bin/activate     # Linux/Mac
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 5. Lancer l'application

```bash
streamlit run app_V2.py
```

---

## 📁 Structure du Projet

```
Corporatebrain/
├── app_V2.py           # Application principale (version courante)
├── app.py              # Version initiale (référence)
├── requirements.txt    # Dépendances Python
├── .streamlit/
│   └── config.toml     # Configuration Streamlit (thème)
├── doc_storage_v2/     # 📂 Dossier des documents (ignoré par git)
└── chroma_db_final_v3/ # 📂 Base vectorielle (générée auto, ignorée par git)
```

> ⚠️ Les documents d'entreprise et la base ChromaDB sont **exclus du dépôt** pour des raisons de confidentialité. Placez vos propres documents dans `doc_storage_v2/` après l'installation.

---

## 📌 Versions

| Version | Fichier | Description |
|---|---|---|
| v1 | `app.py` | Version initiale : upload manuel, ChromaDB basique |
| v2 | `app_V2.py` | Recherche hybride RRF, filtres métadonnées, support DOC/XLSX, sources citées |

---

## 👤 Auteur

Développé par **Zineb** — PFA Atos, 2026
