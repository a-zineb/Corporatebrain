# Corporate Brain — Contexte du projet

## Objectif
Application RAG (Retrieval-Augmented Generation) pour PFA à Atos.
Permet à un employé de poser une question en langage naturel et de recevoir
une réponse générée par IA, basée sur les documents internes de l'entreprise
(actuellement KPSA / MZ), avec les sources citées.

## Résultat attendu
Une application Streamlit fonctionnelle avec :
1. Upload de documents PDF
2. Extraction et découpage du texte (chunking)
3. Génération d'embeddings (Sentence Transformers)
4. Stockage vectoriel (ChromaDB)
5. Recherche sémantique (retrieval)
6. Génération de réponse via LLM local (Ollama / Llama 3)
7. Affichage de la réponse avec ses sources

## Stack technique
- Python
- Streamlit (interface)
- PyMuPDF (extraction PDF)
- Sentence Transformers (embeddings)
- ChromaDB (base vectorielle)
- Ollama + Llama 3 (LLM local)

## Règles importantes
- Toujours expliquer ce que fait un bout de code avant de l'écrire, je suis en train d'apprendre
- Ne jamais tout réécrire d'un coup sans expliquer les changements
- Garder le code simple et lisible, pas besoin d'optimisations complexes
- Le venv doit toujours rester actif avant d'installer quoi que ce soit