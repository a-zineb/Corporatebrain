"""
Outil de Diagnostic Universel RAG - CorporateBrain
Version : 3.0 (Phase 2 - Recherche exhaustive + tableau par document)
Mode    : READ-ONLY - Ne modifie aucun fichier du projet.
Auteur  : Genere par l'agent - base sur commit d823456 de app.py
"""
import sys, io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, io, re, json, argparse
import chromadb
import pandas as pd
import docx
import fitz
import olefile
import ollama
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import rag_forensics

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (doit etre identique a app.py commit d823456)
# ─────────────────────────────────────────────────────────────────────────────
STORAGE_DIR = "doc_storage_v2"
CHROMA_PATH = "chroma_db_local_v2"
COLLECTION  = "documents"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 15
RRF_K       = 60
N_RESULTS   = 10

# Patterns no_documentary_answer alignes avec app.py (14 patterns)
NO_COVERAGE_PATTERNS = [
    r"je ne (trouve|vois|dispose) pas.{0,60}(document|contexte|source|corpus|information)",
    r"n['\u2019]est pas (mentionn|document|trouv|disponible)",
    r"aucune? (information|mention|donn|source|document).{0,60}(document|contexte|corpus|trouv)",
    r"(pas|aucune?) de (r\u00e9ponse|information).{0,50}(contexte|document|source)",
    r"(le |les )?(document|source|corpus|contexte).{0,50}(ne |n['\u2019]).{0,30}(contien|mentionn|trouv|r\u00e9pond|pr\u00e9ci)",
    r"je n['\u2019]ai pas (trouv|d['\u2019]information|acc\u00e8s)",
    r"(impossible|incapable).{0,50}(r\u00e9pondre|trouver|d\u00e9terminer)",
    r"(cette |la )question.{0,50}(hors|au-del\u00e0|d\u00e9passe).{0,30}(p\u00e9rim\u00e8tre|contexte|source|document)",
    r"(bas\u00e9|limit\u00e9|restreint).{0,40}(document|source|contexte|corpus)",
    r"d['\u2019]apr\u00e8s les? (document|source|extrait|contexte).{0,50}(pas|aucun|rien)",
    r"je ne peux pas.{0,50}(confirmer|r\u00e9pondre|dire|pr\u00e9ciser)",
    r"(manque|insuffisan).{0,40}(information|donn|contexte|pr\u00e9cision)",
    r"non (mentionn|r\u00e9pertori|sp\u00e9cifi|d\u00e9taill).{0,40}(document|source|contexte)",
    r"(nothing|no information|not found|not mentioned).{0,60}(document|context|source)",
]

# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────
def sep(char="─", n=70):
    print(char * n)

def header(title, char="═"):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(char * 70)

ICONS = {"PASS": "✓", "FAIL": "✗", "SKIP": "~", "INFO": "i", "WARN": "!"}

def report_step(num, title, status, cause="", proof="", metrics="", recom=""):
    icon = ICONS.get(status, "?")
    print(f"\n[{num:02d}] [{icon}] {title} : {status}")
    if cause:   print(f"       Cause        : {cause}")
    if proof:   print(f"       Preuve       : {proof}")
    if metrics: print(f"       Metriques    : {metrics}")
    if recom:   print(f"       Recommandation : {recom}")


def export_pipeline_trace_report(trace, metrics, expected_behavior, output_dir, case_id):
    """Export a shared trace forensic report without changing standalone tracing."""

    finding = rag_forensics.classify_trace(trace, metrics, expected_behavior)
    if finding.category != "passed":
        rag_forensics.write_failure_report(case_id, finding, trace, output_dir)
    return finding

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DU TEXTE (identique a app.py)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_ole_doc(file_bytes):
    try:
        ole = olefile.OleFileIO(io.BytesIO(file_bytes))
        if not ole.exists('WordDocument'):
            return ''
        word_stream = ole.openstream('WordDocument').read()
        return '\n'.join([
            m.decode('latin-1')
            for m in re.compile(rb'[\x20-\x7E\xC0-\xFF]{4,}').findall(word_stream)
            if m
        ])
    except Exception:
        return ''

def extract_text(file_bytes, file_name):
    ext = os.path.splitext(file_name)[1].lower()
    data = []
    try:
        if ext == ".pdf":
            for i, page in enumerate(fitz.open(stream=file_bytes, filetype="pdf")):
                data.append((f"Page {i+1}", page.get_text()))
        elif ext == ".docx":
            data.append(("Corps du document",
                "\n".join(p.text for p in docx.Document(io.BytesIO(file_bytes)).paragraphs)))
        elif ext == ".doc":
            data.append(("Corps du document", _extract_ole_doc(file_bytes)))
        elif ext == ".xlsx":
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet).fillna("")
                sentences = []
                for _, row in df.iterrows():
                    rt = ", ".join([f"{c}: {v}" for c, v in row.items() if str(v).strip()])
                    if rt:
                        sentences.append(rt + ".")
                data.append((f"Feuille: {sheet}", " ".join(sentences)))
        elif ext == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
            data.append(("Donnees CSV", df.to_string()))
    except Exception:
        pass
    return data

def infer_metadata(filename):
    """Identique a app.py L334-348."""
    normalized = filename.upper()
    entity = "Non classee"
    for e in ["OCM", "OEG", "OJO", "OCI"]:
        if e in normalized:
            entity = e
            break
    app = "Non classee"
    if "KPSA" in normalized:
        app = "KPSA"
    elif "ZM" in normalized or "MZ" in normalized:
        app = "MZ"
    return entity, app

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS FILTRE
# ─────────────────────────────────────────────────────────────────────────────
def meta_passes_filter(meta, chroma_filter):
    """Retourne True si les metadonnees passent le filtre ChromaDB."""
    if not chroma_filter:
        return True
    if "$and" in chroma_filter:
        for cond in chroma_filter["$and"]:
            for k, v in cond.items():
                if meta.get(k) != v:
                    return False
    else:
        for k, v in chroma_filter.items():
            if meta.get(k) != v:
                return False
    return True

def count_chunks_passing_filter(chroma_metas, chroma_filter):
    if not chroma_filter:
        return len(chroma_metas)
    return sum(1 for m in chroma_metas if meta_passes_filter(m, chroma_filter))

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE DE LA REQUETE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_query(query, chroma_docs, bm25):
    """Analyse la requete et sa couverture dans le corpus."""
    header("ANALYSE DE LA REQUETE")
    tokens = query.lower().split()
    acronyms = [t for t in query.split() if re.match(r'^[A-Z]{2,6}$', t)]

    print(f"  Requete      : {query}")
    print(f"  Tokens BM25  : {tokens}")
    print(f"  Acronymes    : {acronyms if acronyms else 'Aucun detecte'}")

    corpus_text = " ".join(chroma_docs).lower()
    print(f"\n  Couverture des tokens dans le corpus :")
    any_absent = False
    for tok in tokens:
        count = corpus_text.count(f" {tok} ") + corpus_text.count(f" {tok},") + corpus_text.count(f" {tok}.")
        if count == 0:
            count = corpus_text.count(tok)
        icon = ICONS["PASS"] if count > 0 else ICONS["FAIL"]
        absent_note = "  <-- ABSENT DU CORPUS" if count == 0 else ""
        print(f"    [{icon}] '{tok}' : ~{count} occurrences{absent_note}")
        if count == 0:
            any_absent = True

    bm25_scores_all = bm25.get_scores(tokens)
    nonzero = sum(1 for s in bm25_scores_all if s > 0)
    print(f"\n  Chunks avec score BM25 > 0 : {nonzero}/{len(bm25_scores_all)}")

    if nonzero == 0:
        print("  [!] AVERTISSEMENT : Aucun chunk ne correspond aux tokens BM25.")
        print("      La recherche BM25 retournera 0 resultats pour cette question.")
        print("      Seule la recherche vectorielle peut trouver des resultats.")
    elif any_absent:
        print("  [!] Certains tokens sont absents du corpus.")
        print("      BM25 ignorera ces tokens (contribue 0 au score).")

# ─────────────────────────────────────────────────────────────────────────────
# CONCLUSION AUTOMATIQUE
# ─────────────────────────────────────────────────────────────────────────────
def generate_conclusion(steps_results):
    """Genere une conclusion en identifiant le premier point de rupture."""
    header("CONCLUSION AUTOMATIQUE")

    failures = [(step, cause) for step, status, cause in steps_results if status == "FAIL"]

    if not failures:
        passes = [step for step, status, _ in steps_results if status == "PASS"]
        print(f"  [✓] Aucune perte d'information detectee ({len(passes)} etapes validees).")
        print()
        print("  Si la reponse est quand meme incorrecte, causes probables :")
        print("    (a) Le LLM ne cite pas le chunk  --> Relancer avec --run-llm")
        print("    (b) Le chunk contient l'info mais elle est noyee dans du bruit")
        print("    (c) La question requiert une inference que le LLM ne fait pas")
        return

    first_step, first_cause = failures[0]
    print(f"  [✗] Point de rupture : etape '{first_step}'")
    print(f"      Cause : {first_cause}")
    print()

    diagnostics = {
        "EXISTENCE":   "-> Le document n'existe pas dans doc_storage_v2/. Uploadez-le via l'interface.",
        "EXTRACTION":  "-> Le fichier est present mais non lisible (PDF scanne? DOCX corrompu? Protection?).",
        "CHUNKING":    "-> Le texte est extrait mais non indexe. Relancez l'ingestion depuis l'interface.",
        "METADONNEES": "-> Le fichier est mal nomme : infer_metadata() ne detecte pas KPSA/MZ/OEG/OCM. Renommez le fichier.",
        "BM25":        "-> Les mots de la requete sont absents du corpus. Reformulez avec les termes exacts des documents.",
        "VECTORIELLE": "-> La semantique de la requete ne correspond pas aux chunks. Reformulez ou ajoutez des synonymes.",
        "RRF":         "-> Document retourne ni par BM25 ni par vectoriel. Reformulez ou verifiez que le document est indexe.",
        "TOP_K":       "-> Document classe trop bas. Le corpus dominant (ex: MBF) occupe les slots RRF. Probleme de bruit documentaire.",
        "PROMPT":      "-> Le repli elargi n'a pas compense le filtre restrictif. Le document est domine meme sans filtre.",
        "LLM":         "-> Chunk fourni au LLM mais non utilise dans la reponse. Probleme de qualite de chunk ou hallucination.",
        "SOURCES":     "-> Reponse bloquee par no_documentary_answer. Le LLM considere ne pas avoir la reponse.",
    }

    for key, msg in diagnostics.items():
        if key in first_step.upper():
            print(f"  {msg}")
            break

    if len(failures) > 1:
        print(f"\n  Autres echecs detectes ({len(failures)-1}) :")
        for step, cause in failures[1:]:
            print(f"    - Etape {step} : {cause}")
# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SUITE (Dataset)
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_SUITE = [
    "Quels sont les flux OEG ?",
    "Comment configurer le KPSA ?",
    "Quelles sont les URLs pour MZ ?",
    "Architecture du réseau OCM",
    "Quel est le port TCP pour KPSA ?",
    "Comment mettre en place une adresse internet statique ?",
    "Procédure pour sécuriser la connexion externe",
    "Je veux un IP fixe pour mon poste",
    "Routage des paquets vers la zone de sécurité",
    "Documentation sur les accès distants",
    "Quel est le nom du président d'Atos ?",
    "Combien coûte une licence Microsoft Office ?",
    "Où se trouve la cantine ?",
    "Quel est le salaire d'un ingénieur réseau ?",
    "Météo à Paris aujourd'hui",
    "How to connect to the secure gateway ?",
    "Coonexion KSPA securisee",
    "KPN seucr access configuration",
    "Configuracion de red OEG",
    "Ip fix",
    "IP",
    "KPSA",
    "OEG",
    "Port",
    "Certificat",
    "Si je perds ma connexion MZ, que dois-je faire ?",
    "Le VPN ne marche plus, aidez-moi",
    "Je n'arrive pas à pinger le serveur OCM",
    "Erreur 403 sur le portail KPSA",
    "Renouvellement de certificat expiré",
    "Quels sont les prérequis pour OEG ?",
    "Comment installer le client VPN KPSA ?",
    "Quelle est l'adresse IP du serveur MZ ?",
    "Protocole de sécurité OCM",
    "Authentification KPSA",
    "Mot de passe admin KPSA",
    "Comment réinitialiser KPSA ?",
    "Où télécharger le client MZ ?",
    "Tutoriel de connexion OEG",
    "Liste des ports bloqués par le firewall",
    "Comment configurer le proxy Atos ?",
    "Quelle est la différence entre OEG et OCM ?",
    "Durée de validité du certificat KPSA",
    "Support niveau 2 pour problème réseau",
    "Comment escalader un ticket réseau ?",
    "VPN lent depuis ce matin",
    "Je suis bloqué sur l'étape 2 du guide KPSA",
    "Est-ce que MZ supporte IPv6 ?",
    "Configuration DNS pour OEG",
    "Procédure de décommissionnement KPSA"
]

# ─────────────────────────────────────────────────────────────────────────────
# OUTIL LLM-AS-A-JUDGE (Validation Scientifique)
# ─────────────────────────────────────────────────────────────────────────────
def llm_as_a_judge(query, chunk_text):
    """
    Appelle Ollama pour verifier si le chunk contient FACTUELLEMENT la reponse.
    Retourne (bool, explication).
    """
    try:
        import ollama
        prompt = f"Tu es un juge strict. Réponds par OUI ou NON uniquement, suivi d'une courte justification. Ce texte contient-il la réponse factuelle à la question ?\nQuestion: {query}\nTexte: {chunk_text}"
        res = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        ans = res["message"]["content"].strip()
        is_yes = ans.upper().startswith("OUI") or "OUI" in ans[:10].upper()
        return is_yes, ans
    except Exception as e:
        return None, f"Erreur du LLM Judge: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONNALITE 6.0 — SCIENTIFIC FORENSIC ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
def generate_forensic_report(query, qv, collection, chroma_filter, id_to_meta, id_to_doc,
                             bm25_all_positive, vec_ids, vec_distances, sorted_rrf, top_k_selected,
                             llm_response, n_results, top_k):
    header("VERDICT SCIENTIFIQUE (RAG FORENSIC)")
    
    category = "ERREUR INTERNE"
    
    # Extraction Mots-cles pour la Preuve Croisee
    stopwords = {"le", "la", "les", "l", "d", "un", "une", "des", "du", "de", "a", "à", "en", "dans", "pour", "par", "sur", "avec", "sans", "sous", "comment", "quoi", "quel", "quelle", "quels", "quelles", "qui", "est", "ce", "que", "pourquoi", "et", "ou", "ni", "mais", "donc", "or", "car"}
    tokens = [t.lower() for t in query.replace("'", " ").replace(".", " ").replace(",", " ").split()]
    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]

    # Ground Truth Semantique
    try:
        vr_exhaustive = collection.query(
            query_embeddings=[qv],
            n_results=3,
            where=chroma_filter,
            include=["distances"]
        )
        gt_ids = vr_exhaustive.get("ids", [[]])[0]
        gt_dists = vr_exhaustive.get("distances", [[]])[0]
    except Exception as e:
        print(f"  [ERREUR] Impossible d'interroger ChromaDB pour le diagnostic : {e}")
        return "ERREUR INTERNE"

    if not gt_ids:
        print("[FAITS DÉMONTRÉS]")
        print("- Le filtre UI rejette absolument tous les documents de la base.")
        print("\n[LIMITES INTRINSÈQUES / CAUSES RACINES]")
        print("- Pipeline bloqué avant même la recherche.")
        print("\n[RECOMMANDATION]")
        print("- Désactiver ou élargir le filtre.")
        return "FILTRE BLOQUANT"

    SEUIL_VIDE = 1.2
    best_dist = gt_dists[0]
    target_cid = gt_ids[0]
    
    # Preuve Croisee: Mots-cles vs Sémantique
    doc_text = id_to_doc.get(target_cid, "")
    doc_lower = doc_text.lower()
    kw_found = sum(1 for kw in keywords if kw in doc_lower) if keywords else 0
    
    if best_dist > SEUIL_VIDE:
        print("[FAITS DÉMONTRÉS]")
        print(f"- Le document sémantiquement le plus proche a une distance de {best_dist:.3f} (Seuil de rejet: {SEUIL_VIDE}).")
        if kw_found == 0:
            print("- Les mots-clés exacts sont également introuvables.")
        
        print("\n[HYPOTHÈSES ÉCARTÉES]")
        print("- Écarté : Problème de retrieval (L'information n'existe tout simplement pas).")
        
        print("\n[LIMITES INTRINSÈQUES / CAUSES RACINES]")
        if kw_found > 0:
            print("- Le corpus contient les mots-clés, mais le contexte sémantique est totalement différent (Hors-Sujet prouvé).")
        else:
            print("- L'information demandée est physiquement absente du corpus.")
            
        print("\n[RECOMMANDATION]")
        print("- Uploader un document contenant cette information.")
        return "CORPUS VIDE"

    target_fname = id_to_meta.get(target_cid, {}).get("source_file", "?")
    
    # Lifecycle
    bm25_entry = next(((r, score) for r, (cid, score, _) in enumerate(bm25_all_positive) if cid == target_cid), None)
    bm25_rank = bm25_entry[0] if bm25_entry else None
    
    vec_entry = next(((r, dist) for r, (cid, dist) in enumerate(zip(vec_ids, vec_distances)) if cid == target_cid), None)
    vec_rank  = vec_entry[0] if vec_entry else None
    
    rrf_entry = next(((r, score) for r, (cid, score) in enumerate(sorted_rrf) if cid == target_cid), None)
    rrf_rank  = rrf_entry[0] if rrf_entry else None
    
    in_top_k = any(cid == target_cid for cid, _ in top_k_selected)
    
    faits = []
    hypo = []
    limites = []
    recom = ""
    
    if len(gt_ids) > 1 and gt_dists[1] < SEUIL_VIDE:
        faits.append("PLUSIEURS CHUNKS pertinents ont été identifiés. L'analyse se concentre sur le meilleur.")
        
    faits.append(f"L'information exacte cible le chunk '{target_cid}' (Fichier : {target_fname}).")
    faits.append(f"Preuve sémantique : Distance excellente ({best_dist:.3f} < {SEUIL_VIDE}).")
    if kw_found > 0:
        faits.append(f"Preuve textuelle : Contient {kw_found}/{len(keywords)} mots-clés.")
    else:
        faits.append(f"Preuve textuelle : 0 mot-clé trouvé (Décalage de vocabulaire surmonté par le modèle vectoriel).")
        
    faits.append(f"Ce chunk a passé BM25 (Rang {bm25_rank if bm25_rank is not None else 'Rejeté'}) et Vectoriel (Rang {vec_rank if vec_rank is not None else 'Rejeté'}).")
    
    if bm25_rank is None and vec_rank is None:
        hypo.append("Écarté : Problème d'indexation (Le chunk existe et est valide isolément).")
        limites.append("Le chunk contenant la réponse a été exclu avant même le RRF, probablement par un filtre UI.")
        recom = "Vérifier les filtres Streamlit appliqués sur ce document."
        category = "FILTRE BLOQUANT"
    elif bm25_rank is not None and bm25_rank >= n_results:
        hypo.append("Écarté : Problème sémantique (Le chunk répond bien à la question).")
        limites.append(f"Le chunk est éliminé par la limite de récupération BM25 (Rang {bm25_rank} >= {n_results}). Le bruit documentaire statistique étouffe la vérité.")
        recom = "Augmenter n_results, ou utiliser des mots-clés beaucoup plus discriminants."
        category = "BRUIT BM25"
    elif vec_rank is not None and vec_rank >= n_results:
        hypo.append("Écarté : Problème de chunking (Le chunk est autonome).")
        limites.append(f"Le chunk est noyé dans le bruit sémantique (Rang Vec {vec_rank} >= {n_results}). La fenêtre de récupération vectorielle est trop étroite.")
        recom = "Le modèle d'embedding juge ce chunk moins pertinent que le bruit. Améliorer la question ou augmenter n_results."
        category = "BRUIT VECTORIEL"
    elif rrf_rank is not None and rrf_rank >= top_k:
        hypo.append("Écarté : Problème de retrieval individuel (Le chunk a survécu à BM25 et Vectoriel).")
        limites.append(f"Le chunk a survécu jusqu'à la fusion, mais a fini au rang {rrf_rank} >= top_k ({top_k}). Bloqué par la limite top_k.")
        recom = f"Augmenter top_k à {rrf_rank + 1} minimum, ou nettoyer le corpus des documents parasites."
        category = "BRUIT RRF"
    elif in_top_k:
        faits.append(f"Ce chunk a été retenu dans le Top-K et envoyé au LLM.")
        hypo.append("Écarté : Problème de filtrage ou de Retrieval (Le pipeline a parfaitement fonctionné).")
        
        # LLM AS A JUDGE
        print("  [i] Appel discret au LLM-Judge pour valider factuellement le chunk...")
        is_yes, judge_reason = llm_as_a_judge(query, doc_text)
        
        if is_yes is True:
            faits.append(f"Test LLM-Judge : Le chunk contient factuellement la réponse (Juge dit : {judge_reason}).")
            if llm_response and "no_documentary_answer" not in llm_response.lower() and "je ne dispose pas" not in llm_response.lower():
                limites.append("L'IA principale a lu le chunk mais a formulé une réponse insatisfaisante (Hallucination probable).")
                recom = "Réduire la température de l'IA ou utiliser un modèle plus performant."
                category = "HALLUCINATION LLM"
            else:
                limites.append("L'IA principale a reçu la réponse mais a déclenché le rejet 'no_documentary_answer'. Le modèle manque de capacité de raisonnement sur ce prompt.")
                recom = "Améliorer le prompt système pour forcer l'IA à extraire l'information sans douter."
                category = "REJET LLM (Capacité insuffisante)"
        elif is_yes is False:
            faits.append(f"Test LLM-Judge : Le chunk est INCOMPLET ou HORS-SUJET par rapport à la question factuelle (Juge dit : {judge_reason}).")
            limites.append("Bien que sémantiquement le plus proche, le chunk ne contient pas la réponse factuelle exploitable. Troncature probable par l'algorithme de chunking.")
            recom = "Augmenter la taille des chunks (chunk_size) ou le chevauchement (overlap) pour ne pas couper les phrases clés."
            category = "CHUNK INCOMPLET"
        else:
            limites.append(f"Impossible de juger le chunk via l'IA ({judge_reason}).")
            recom = "Vérifier manuellement le chunk."
            category = "NON JUGEABLE"
    else:
        limites.append("Comportement anormal du pipeline.")
        recom = "Inspecter les logs détaillés du pipeline RRF."
        category = "ERREUR INTERNE"
        
    print("\n[FAITS DÉMONTRÉS]")
    for f in faits:
        print(f"- {f}")
        
    print("\n[HYPOTHÈSES ÉCARTÉES]")
    for h in hypo:
        print(f"- {h}")
        
    print("\n[LIMITES INTRINSÈQUES / CAUSES RACINES]")
    for l in limites:
        print(f"- {l}")
        
    print("\n[RECOMMANDATION]")
    print(f"- {recom}")
    
    return category


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONNALITE A — RECHERCHE EXHAUSTIVE (--exhaustive)
# ─────────────────────────────────────────────────────────────────────────────
def exhaustive_search(qv, bscores, sidx, collection, chroma_metas, chroma_ids,
                      idx_to_id, id_to_meta, chroma_filter, efile, total_chunks):
    """
    Calcule le rang REEL d'un fichier attendu dans TOUT le corpus.
    Distingue : information absente (score BM25 = 0) vs information mal classee
    (score > 0 mais rang > n_results et eliminee avant le RRF).
    Necessite --exhaustive. Optionnel car plus lent.
    """
    header("RECHERCHE EXHAUSTIVE — Rang reel dans tout le corpus")
    print(f"  [i] Corpus total analyse : {total_chunks} chunks")

    # ── BM25 exhaustif ────────────────────────────────────────────────────────
    all_positive_count = sum(1 for s in bscores if s > 0)
    print(f"  BM25 : {all_positive_count}/{total_chunks} chunks ont un score > 0")

    if efile:
        print(f"\n  Fichier cible : {efile}")
        sep("─", 55)

        # Rang BM25 global (sur tout le corpus, sans limite n_results)
        bm25_global_rank   = None
        bm25_global_score  = None
        bm25_filtered_rank = None
        filtered_counter   = 0

        for global_rank, i in enumerate(sidx):
            if bscores[i] <= 0:
                break
            cid    = idx_to_id[i]
            meta   = chroma_metas[i]
            passes = meta_passes_filter(meta, chroma_filter)

            if meta.get("source_file") == efile and bm25_global_rank is None:
                bm25_global_rank  = global_rank
                bm25_global_score = bscores[i]
                bm25_filtered_rank = filtered_counter if passes else None

            if passes:
                filtered_counter += 1

        # Affichage BM25
        if bm25_global_rank is not None:
            icon = ICONS["PASS"] if bm25_global_rank < N_RESULTS else ICONS["WARN"]
            print(f"  [{icon}] BM25 rang global (corpus entier) : {bm25_global_rank} | Score : {bm25_global_score:.4f}")
            if chroma_filter:
                if bm25_filtered_rank is not None:
                    icon2 = ICONS["PASS"] if bm25_filtered_rank < N_RESULTS else ICONS["WARN"]
                    print(f"  [{icon2}] BM25 rang apres filtre          : {bm25_filtered_rank}")
                else:
                    print(f"  [✗] Ce chunk est rejete par le filtre (metadata incompatible)")
            if bm25_global_rank >= N_RESULTS:
                print(f"\n  -> DIAGNOSTIC : Le fichier EXISTE dans BM25 au rang {bm25_global_rank}.")
                print(f"     Mais n_results={N_RESULTS} l'elimine avant que le RRF puisse l'evaluer.")
                print(f"     L'information N'EST PAS ABSENTE du corpus.")
                print(f"     Elle est MAL CLASSEE : la requete ne correspond pas assez aux termes du document.")
                confidence = max(10, min(95, 100 - (bm25_global_rank * 2)))
                print(f"     Confiance du diagnostic : ~{confidence}%")
        else:
            # Verifier si le fichier a des chunks dans ChromaDB
            efile_chunk_ids = [cid for cid, meta in id_to_meta.items()
                               if meta.get("source_file") == efile]
            if efile_chunk_ids:
                print(f"  [✗] BM25 rang global : NON TROUVE (score = 0.000 pour tous les chunks)")
                print(f"\n  -> DIAGNOSTIC : Le fichier est indexe ({len(efile_chunk_ids)} chunks dans ChromaDB)")
                print(f"     mais AUCUN token de la requete ne correspond au contenu de ses chunks.")
                print(f"     Les mots de la question sont ABSENTS du document (ou de son vocabulary BM25).")
                print(f"     Confiance du diagnostic : ~95%")
            else:
                print(f"  [✗] BM25 rang global : FICHIER NON INDEXE (0 chunk dans ChromaDB)")

        # ── Vectoriel exhaustif (top 100) ─────────────────────────────────────
        n_exhaustive = min(100, total_chunks)
        print(f"\n  Recherche vectorielle etendue (top {n_exhaustive} sur {total_chunks})...")
        try:
            vr_ex      = collection.query(
                query_embeddings=[qv],
                n_results=n_exhaustive,
                include=["metadatas"]
            )
            vec_ex_ids   = vr_ex.get("ids", [[]])[0]
            vec_ex_metas = vr_ex.get("metadatas", [[]])[0]

            vec_global_rank = None
            for rank, (cid, meta) in enumerate(zip(vec_ex_ids, vec_ex_metas)):
                if meta.get("source_file") == efile:
                    vec_global_rank = rank
                    break

            if vec_global_rank is not None:
                icon = ICONS["PASS"] if vec_global_rank < N_RESULTS else ICONS["WARN"]
                print(f"  [{icon}] Vectoriel rang (top {n_exhaustive}) : {vec_global_rank}")
                if vec_global_rank >= N_RESULTS:
                    print(f"\n  -> DIAGNOSTIC : Le fichier est detecte par vectoriel au rang {vec_global_rank}.")
                    print(f"     Mais n_results={N_RESULTS} l'elimine avant que le RRF puisse l'evaluer.")
                    print(f"     La requete et le document sont semantiquement proches mais pas assez.")
                    confidence = max(10, min(90, 100 - (vec_global_rank * 1.5)))
                    print(f"     Confiance du diagnostic : ~{confidence:.0f}%")
            else:
                print(f"  [✗] Vectoriel rang : NON TROUVE dans le top {n_exhaustive}")
                print(f"\n  -> DIAGNOSTIC : Le fichier n'est pas semantiquement similaire a la requete.")
                print(f"     La recherche vectorielle ne peut pas le retrouver, meme avec un top plus grand.")
                print(f"     Confiance du diagnostic : ~80%")
        except Exception as e:
            print(f"  [!] Erreur requete vectorielle etendue : {e}")

    else:
        # Sans fichier attendu : distribution des scores BM25 dans le corpus
        print(f"\n  Distribution BM25 globale (tous les chunks) :")
        sep("─", 55)
        buckets = [
            ("Score = 0       ", lambda s: s <= 0),
            ("0 < Score <= 1  ", lambda s: 0 < s <= 1),
            ("1 < Score <= 5  ", lambda s: 1 < s <= 5),
            ("5 < Score <= 10 ", lambda s: 5 < s <= 10),
            ("Score > 10      ", lambda s: s > 10),
        ]
        for label, condition in buckets:
            cnt = sum(1 for s in bscores if condition(s))
            pct = 100 * cnt // total_chunks
            bar = "█" * (pct // 3)
            print(f"    {label} : {cnt:4d} chunks ({pct:3d}%) {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONNALITE B — TABLEAU RECAPITULATIF PAR DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
def build_document_table(bm25_all_positive, vec_ids, vec_metas, sorted_rrf,
                         top_k_ids, id_to_meta, chroma_metas, efile=None):
    """
    Construit un tableau consolide PAR FICHIER SOURCE (pas par chunk).
    Pour chaque fichier apparu dans les resultats, affiche :
    nb chunks totaux | meilleur rang BM25 | meilleur rang vectoriel |
    meilleur rang RRF | dans top_k | note de diagnostic.
    """
    header("TABLEAU RECAPITULATIF PAR DOCUMENT SOURCE")

    # ── Construction des donnees par fichier ──────────────────────────────────
    files_data = {}  # fname -> dict

    def get_or_create(fname):
        if fname not in files_data:
            files_data[fname] = {
                "bm25_best": None, "bm25_score": None, "bm25_chunks": 0,
                "vec_best": None,
                "rrf_best": None, "rrf_score": None,
                "in_topk": False,
            }
        return files_data[fname]

    # BM25 (liste complete avec score > 0, sans limite n_results)
    for rank, (cid, score, meta) in enumerate(bm25_all_positive):
        fname = meta.get("source_file", "?")
        d = get_or_create(fname)
        d["bm25_chunks"] += 1
        if d["bm25_best"] is None or rank < d["bm25_best"]:
            d["bm25_best"]  = rank
            d["bm25_score"] = score

    # Vectoriel
    for rank, (cid, meta) in enumerate(zip(vec_ids, vec_metas)):
        fname = meta.get("source_file", "?")
        d = get_or_create(fname)
        if d["vec_best"] is None or rank < d["vec_best"]:
            d["vec_best"] = rank

    # RRF et top_k
    for rank, (cid, rrf_score) in enumerate(sorted_rrf):
        meta  = id_to_meta.get(cid, {})
        fname = meta.get("source_file", "?")
        d = get_or_create(fname)
        if d["rrf_best"] is None or rank < d["rrf_best"]:
            d["rrf_best"]  = rank
            d["rrf_score"] = rrf_score
        if cid in top_k_ids:
            d["in_topk"] = True

    # Nombre total de chunks par fichier dans ChromaDB
    chunks_per_file = {}
    for meta in chroma_metas:
        fname = meta.get("source_file", "?")
        chunks_per_file[fname] = chunks_per_file.get(fname, 0) + 1

    # Ajouter le fichier attendu s'il n'est pas apparu dans les resultats
    if efile and efile not in files_data:
        if efile in chunks_per_file:
            get_or_create(efile)  # entree vide = tous les rangs a None

    # ── Tri : meilleur rang RRF d'abord, puis les autres ─────────────────────
    def sort_key(item):
        _, d = item
        rrf = d.get("rrf_best")
        return (0 if rrf is not None else 1, rrf if rrf is not None else 9999)

    sorted_files = sorted(files_data.items(), key=sort_key)

    # ── Affichage ─────────────────────────────────────────────────────────────
    W = 48  # largeur colonne nom de fichier
    print(f"\n  {'FICHIER':<{W}} | TOT | BM25 | VEC | RRF | TOP_K | DIAGNOSTIC")
    print(f"  {'─'*W}─+─────+──────+─────+─────+───────+─────────────────────────")

    for fname, d in sorted_files:
        total_f  = str(chunks_per_file.get(fname, "?"))
        bm25_s   = f"{d['bm25_best']:4d}" if d["bm25_best"] is not None else "   -"
        vec_s    = f"{d['vec_best']:4d}"  if d["vec_best"]  is not None else "   -"
        rrf_s    = f"{d['rrf_best']:4d}"  if d["rrf_best"]  is not None else "   -"
        topk_s   = " OUI " if d["in_topk"] else " NON "

        # Note de diagnostic automatique
        if d["rrf_best"] is None and d["bm25_best"] is None and d["vec_best"] is None:
            if chunks_per_file.get(fname, 0) > 0:
                note = "ABSENT DES RESULTATS (tokens absents du doc)"
            else:
                note = "NON INDEXE"
        elif d["in_topk"]:
            note = "DANS LE PROMPT"
        elif d["rrf_best"] is not None and d["rrf_best"] >= TOP_K:
            note = f"PERDU top_k : rang RRF {d['rrf_best']} >= {TOP_K}"
        elif d["bm25_best"] is not None and d["bm25_best"] >= N_RESULTS:
            note = f"BM25 rang {d['bm25_best']} > n_results={N_RESULTS}"
        elif d["vec_best"] is not None and d["bm25_best"] is None:
            note = "Vectoriel seul (BM25 score=0)"
        elif d["bm25_best"] is not None and d["vec_best"] is None:
            note = "BM25 seul (absent vectoriel)"
        else:
            note = ""

        mark = " <-- ATTENDU" if efile and fname == efile else ""
        display = fname[:W-1] if len(fname) > W else fname
        print(f"  {display:<{W}} | {total_f:>3s} | {bm25_s} | {vec_s} | {rrf_s} | {topk_s} | {note}{mark}")

    # Legende
    print(f"\n  Legende : TOT=chunks dans ChromaDB | BM25/VEC/RRF=meilleur rang | TOP_K=inclu dans prompt")
    print(f"            Rangs affiches sur le filtrage actif | - = non retrouve")

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONNALITE 3.1 — RECHERCHE PLEIN TEXTE
# ─────────────────────────────────────────────────────────────────────────────
def fulltext_search(search_term, chroma_docs, id_to_meta, chroma_ids):
    header(f"RECHERCHE PLEIN TEXTE EXACTE : '{search_term}'")
    term = search_term.lower()
    
    results = []
    for cid, doc in zip(chroma_ids, chroma_docs):
        doc_lower = doc.lower()
        if term in doc_lower:
            meta = id_to_meta.get(cid, {})
            fname = meta.get("source_file", "?")
            idx = doc_lower.find(term)
            start = max(0, idx - 40)
            end = min(len(doc), idx + len(term) + 40)
            snippet = doc[start:end].replace('\n', ' ')
            results.append((fname, snippet))
            
    if not results:
        print(f"  [✗] Le terme '{search_term}' n'apparait STRICTEMENT NULLE PART dans les {len(chroma_docs)} chunks.")
        print("      L'information est physiquement absente du corpus indexe.")
    else:
        print(f"  [✓] Terme trouve dans {len(results)} chunk(s) :")
        by_file = {}
        for fname, snip in results:
            if fname not in by_file:
                by_file[fname] = []
            by_file[fname].append(snip)
            
        for fname, snippets in by_file.items():
            print(f"    - {fname} ({len(snippets)} occurrence(s))")
            for i, snip in enumerate(snippets[:2]):
                print(f"        ...{snip}...")
            if len(snippets) > 2:
                print(f"        (+ {len(snippets)-2} autres occurrences)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Outil de diagnostic universel du pipeline RAG CorporateBrain v3.0"
    )
    parser.add_argument("--query",         type=str, default=None,
                        help="Question utilisateur a diagnostiquer (requis hors mode validation)")
    parser.add_argument("--validation-suite", action="store_true",
                        help="Executer le batch complet de 50 questions de validation")
    parser.add_argument("--expected-file", type=str, default=None,
                        help="(Optionnel) Nom du fichier censé contenir la reponse")
    parser.add_argument("--filter",        type=str, default=None,
                        help='Filtre JSON (ex: \'{"application": "KPSA"}\')')
    parser.add_argument("--top-n",         type=int, default=15,
                        help="Nombre de resultats a afficher dans les classements (defaut: 15)")
    parser.add_argument("--exhaustive",    action="store_true",
                        help="Recherche le rang reel dans TOUT le corpus (pas seulement top 10)")
    parser.add_argument("--search-text",   type=str, default=None,
                        help="Effectuer une recherche plein texte independante du retrieval")
    parser.add_argument("--show-chunks",   type=int, default=0,
                        help="Afficher les N premiers caracteres de chaque chunk du prompt")
    parser.add_argument("--investigation", action="store_true",
                        help="Mode investigation : diagnostic complet et autopsie automatique")
    parser.add_argument("--run-llm",       action="store_true",
                        help="Executer les etapes 10-11 avec Ollama (plus lent)")
    args = parser.parse_args()
    
    if not args.query and not args.validation_suite:
        parser.error("L'argument --query est requis, a moins d'utiliser --validation-suite.")

    if args.validation_suite:
        args.investigation = True
        args.run_llm = True
    elif args.investigation:
        args.exhaustive = True
        args.run_llm = True

    header("DIAGNOSTIC RAG TRACER v3.2")
    print(f"  Question        : {q}")
    if args.investigation:
        print("  [!] MODE INVESTIGATION ACTIF : Recherches exhaustives et appel LLM forces.")
    print(f"  Fichier attendu : {efile if efile else '(non specifie - mode universel)'}")
    print(f"  Filtre UI       : {chroma_filter if chroma_filter else 'Aucun'}")
    print(f"  Top-N affiche   : {top_n}")
    q             = args.query
    efile         = args.expected_file
    top_n         = args.top_n
    chroma_filter = json.loads(args.filter) if args.filter else None

def run_single_trace(q, args, collection, embedding_model, bm25, chroma_metas, chroma_ids, chroma_docs, id_to_meta, id_to_doc, idx_to_id, total_chunks):
    efile         = args.expected_file
    top_n         = args.top_n
    chroma_filter = json.loads(args.filter) if args.filter else None
    
    header("DIAGNOSTIC RAG TRACER v3.2")

    embedding_model = SentenceTransformer(EMBED_MODEL)

    all_data     = collection.get(include=["documents", "metadatas"])
    chroma_docs  = all_data["documents"]
    chroma_metas = all_data["metadatas"]
    chroma_ids   = all_data["ids"]

    id_to_meta = {cid: meta for cid, meta in zip(chroma_ids, chroma_metas)}
    id_to_doc  = {cid: doc  for cid, doc  in zip(chroma_ids, chroma_docs)}
    idx_to_id  = {i: cid    for i, cid    in enumerate(chroma_ids)}

    bm25 = BM25Okapi([d.lower().split() for d in chroma_docs])

    print(f"  ChromaDB : {total_chunks} chunks indexes")
    print(f"  BM25     : {total_chunks} documents indexes")
    print(f"  Modele   : {EMBED_MODEL}")

    if chroma_filter:
        filtered_count = count_chunks_passing_filter(chroma_metas, chroma_filter)
        print(f"  Filtre actif : {filtered_count}/{total_chunks} chunks passent")
        if filtered_count == 0:
            print("  [!] AVERTISSEMENT : Aucun chunk ne passe ce filtre.")
            print("      Toute recherche filtree retournera 0 resultats.")

    steps_results = []
    stop_trace    = False

    # ─── ANALYSE DE LA REQUETE ────────────────────────────────────────────────
    analyze_query(q, chroma_docs, bm25)

    if args.search_text:
        fulltext_search(args.search_text, chroma_docs, id_to_meta, chroma_ids)

    # ─── ETAPES 1-4 : TRACE DU FICHIER ATTENDU (si fourni) ──────────────────
    if efile:
        header("TRACE DU FICHIER ATTENDU — Etapes 1 a 4")
        fpath = os.path.join(STORAGE_DIR, efile)

        if not os.path.exists(fpath):
            report_step(1, "EXISTENCE DU DOCUMENT", "FAIL",
                cause=f"'{efile}' introuvable dans {STORAGE_DIR}/",
                recom="Uploadez le fichier via l'interface Streamlit")
            steps_results.append(("EXISTENCE", "FAIL", f"'{efile}' absent du stockage"))
            stop_trace = True
        else:
            size_kb = os.path.getsize(fpath) / 1024
            report_step(1, "EXISTENCE DU DOCUMENT", "PASS",
                proof=f"Taille : {size_kb:.1f} KB")
            steps_results.append(("EXISTENCE", "PASS", ""))

            try:
                fbytes    = open(fpath, "rb").read()
                extracted = extract_text(fbytes, efile)
                total_chars = sum(len(t) for _, t in extracted)
                if total_chars > 0:
                    report_step(2, "EXTRACTION DU TEXTE", "PASS",
                        proof=f"{total_chars} chars extraits en {len(extracted)} section(s)")
                    steps_results.append(("EXTRACTION", "PASS", ""))
                else:
                    report_step(2, "EXTRACTION DU TEXTE", "FAIL",
                        cause="0 caractere extrait (PDF scanne, DOCX corrompu, ou protection)",
                        recom="Verifiez le format du fichier et ouvrez-le manuellement")
                    steps_results.append(("EXTRACTION", "FAIL", "Extraction vide (0 chars)"))
                    stop_trace = True
            except Exception as e:
                report_step(2, "EXTRACTION DU TEXTE", "FAIL", cause=str(e))
                steps_results.append(("EXTRACTION", "FAIL", str(e)))
                stop_trace = True

        if not stop_trace:
            efile_chunk_ids = [cid for cid, meta in id_to_meta.items()
                               if meta.get("source_file") == efile]
            if not efile_chunk_ids:
                report_step(3, "CHUNKING ET INDEXATION", "FAIL",
                    cause="Texte extrait mais aucun chunk trouve dans ChromaDB",
                    recom="Relancez l'ingestion depuis l'interface")
                steps_results.append(("CHUNKING", "FAIL", "0 chunk dans ChromaDB"))
                stop_trace = True
            else:
                sizes      = [len(id_to_doc.get(cid, "")) for cid in efile_chunk_ids]
                avg_size   = sum(sizes) / len(sizes) if sizes else 0
                long_chunks = sum(1 for s in sizes if s > 1000)
                report_step(3, "CHUNKING ET INDEXATION", "PASS",
                    proof=f"{len(efile_chunk_ids)} chunks | moy={avg_size:.0f} chars | min={min(sizes)} | max={max(sizes)}",
                    metrics=f"Chunks > 1000 chars : {long_chunks}/{len(sizes)} ({100*long_chunks//len(sizes)}%)")
                steps_results.append(("CHUNKING", "PASS", ""))

        if not stop_trace:
            entity, app = infer_metadata(efile)
            if chroma_filter:
                passes = meta_passes_filter({"application": app, "geographical_entity": entity}, chroma_filter)
                if passes:
                    report_step(4, "METADONNEES ET FILTRE", "PASS",
                        proof=f"Tag infere : app={app}, entity={entity} -> passe le filtre")
                    steps_results.append(("METADONNEES", "PASS", ""))
                else:
                    report_step(4, "METADONNEES ET FILTRE", "FAIL",
                        cause=f"Tag infere : app={app}, entity={entity} | Filtre attend : {chroma_filter}",
                        proof=f"Regle infer_metadata : nom='{efile}' -> app='{app}', entity='{entity}'",
                        recom="Le nom du fichier ne contient pas les tokens reconnus (KPSA, MZ, OCM, OEG...)")
                    steps_results.append(("METADONNEES", "FAIL", f"Tag ({app}/{entity}) incompatible avec filtre"))
                    stop_trace = True
            else:
                report_step(4, "METADONNEES ET FILTRE", "SKIP",
                    proof=f"Tag infere : app={app}, entity={entity} | Aucun filtre UI actif")
                steps_results.append(("METADONNEES", "SKIP", ""))
    else:
        print("\n  [i] Mode universel : etapes 1-4 ignorees (aucun fichier attendu specifie).")
        print("      Ajoutez --expected-file 'nom_du_fichier.ext' pour tracer un document precis.")

    # ─── ETAPES 5-8 : RETRIEVAL ──────────────────────────────────────────────
    sorted_rrf = []  # toujours initialise
    top_k_ids  = set()
    bm25_all_positive = []
    vec_ids, vec_metas = [], []
    bm25_results  = []
    bm25_rejected = []
    bm25_ids_set  = set()
    vec_ids_set   = set()
    qv            = None

    if not stop_trace:
        header("RETRIEVAL — Etapes 5 a 8")

        qv = embedding_model.encode(q).tolist()

        # --- Recherche vectorielle ---
        n_vec = min(N_RESULTS, total_chunks)
        try:
            vr = collection.query(
                query_embeddings=[qv],
                n_results=n_vec,
                where=chroma_filter,
                include=["documents", "metadatas", "distances"]
            )
            vec_ids       = vr.get("ids", [[]])[0]
            vec_metas     = vr.get("metadatas", [[]])[0]
            vec_distances = vr.get("distances", [[]])[0]
        except Exception as e:
            vec_ids, vec_metas, vec_distances = [], [], []
            print(f"  [!] Erreur ChromaDB vectoriel : {e}")

        # --- BM25 ---
        tq      = q.lower().split()
        bscores = bm25.get_scores(tq)
        sidx    = sorted(range(len(bscores)), key=lambda i: bscores[i], reverse=True)

        for i in sidx:
            if bscores[i] <= 0:
                break
            bm25_all_positive.append((idx_to_id[i], bscores[i], chroma_metas[i]))

        for cid, score, meta in bm25_all_positive:
            passes = meta_passes_filter(meta, chroma_filter)
            if not passes:
                bm25_rejected.append((cid, f"Score={score:.3f} - rejete par filtre (app={meta.get('application')}, entity={meta.get('geographical_entity')})", meta))
                continue
            if len(bm25_results) >= N_RESULTS:
                bm25_rejected.append((cid, f"Score={score:.3f} - rang > n_results={N_RESULTS}", meta))
                continue
            bm25_results.append((cid, score, meta))

        bm25_ids_set = set(cid for cid, _, _ in bm25_results)
        vec_ids_set  = set(vec_ids)
        both_ids     = bm25_ids_set & vec_ids_set

        # --- Fusion RRF ---
        rrf_scores = {}
        for rank, cid in enumerate(vec_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rank + 1 + RRF_K)
        for rank, (cid, sc, meta) in enumerate(bm25_results):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rank + 1 + RRF_K)
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Classement BM25 ───────────────────────────────────────────────────
        print(f"\n[05] RECHERCHE BM25")
        sep()
        print(f"  {len(bm25_results)} resultats retenus | {len(bm25_rejected)} rejetes | affichage top {min(top_n, len(bm25_results))}")
        print()
        if bm25_results:
            for rank, (cid, score, meta) in enumerate(bm25_results[:top_n]):
                fname = meta.get("source_file", "?")[:55]
                app   = meta.get("application", "?")
                mark  = " <-- FICHIER ATTENDU" if efile and meta.get("source_file") == efile else ""
                print(f"  Rang {rank:2d} | Score {score:6.3f} | {app:15s} | {fname}{mark}")
            steps_results.append(("BM25", "PASS", f"Top1={bm25_results[0][2].get('source_file','?')}"))
        else:
            print("  Aucun resultat BM25 (score = 0 pour tous les chunks)")
            steps_results.append(("BM25", "FAIL", "Score BM25 = 0 pour tous les chunks filtres"))

        if bm25_rejected:
            print(f"\n  Chunks rejetes BM25 ({len(bm25_rejected)} total, affichage max 5) :")
            for cid, reason, meta in bm25_rejected[:5]:
                print(f"    {meta.get('source_file','?')[:50]} | {reason}")

        if efile:
            efile_in_bm25       = [r for r, (cid, sc, m) in enumerate(bm25_results) if m.get("source_file") == efile]
            efile_rejected_bm25 = [(reason, m) for cid, reason, m in bm25_rejected if m.get("source_file") == efile]
            if not efile_in_bm25 and efile_rejected_bm25:
                print(f"\n  [!] Fichier attendu REJETE par BM25 : {efile_rejected_bm25[0][0]}")

        # ── Classement vectoriel ──────────────────────────────────────────────
        print(f"\n[06] RECHERCHE VECTORIELLE")
        sep()
        print(f"  {len(vec_ids)} resultats | affichage top {min(top_n, len(vec_ids))}")
        print()
        if vec_ids:
            for rank, (cid, meta) in enumerate(zip(vec_ids[:top_n], vec_metas[:top_n])):
                fname = meta.get("source_file", "?")[:55]
                app   = meta.get("application", "?")
                mark  = " <-- FICHIER ATTENDU" if efile and meta.get("source_file") == efile else ""
                print(f"  Rang {rank:2d} | {app:15s} | {fname}{mark}")
            steps_results.append(("VECTORIELLE", "PASS", f"Top1={vec_metas[0].get('source_file','?') if vec_metas else '?'}"))
        else:
            print("  Aucun resultat vectoriel (filtre trop restrictif ?)")
            steps_results.append(("VECTORIELLE", "FAIL", "Aucun resultat vectoriel"))

        # ── Classement RRF ────────────────────────────────────────────────────
        print(f"\n[07] FUSION RRF")
        sep()
        print(f"  {len(sorted_rrf)} candidats totaux")
        print(f"  Retrouves par les DEUX (BM25+Vec) : {len(both_ids)}")
        print(f"  Vectoriel seul : {len(vec_ids_set - bm25_ids_set)} | BM25 seul : {len(bm25_ids_set - vec_ids_set)}")
        print()
        print(f"  Legende : [V.]=vectoriel seul | [.B]=BM25 seul | [VB]=les deux")
        print()

        efile_rrf_rank = None
        for rank, (cid, rrf_score) in enumerate(sorted_rrf[:top_n]):
            meta   = id_to_meta.get(cid, {})
            fname  = meta.get("source_file", "?")[:52]
            app    = meta.get("application", "?")
            in_vec = "V" if cid in vec_ids_set  else "."
            in_bm  = "B" if cid in bm25_ids_set else "."
            vec_rank_str  = next((str(r) for r, cid2 in enumerate(vec_ids) if cid2 == cid), "-")
            bm25_rank_str = next((str(r) for r, (cid2, _, _) in enumerate(bm25_results) if cid2 == cid), "-")
            mark = " <-- FICHIER ATTENDU" if efile and meta.get("source_file") == efile else ""
            print(f"  Rang {rank:2d} | RRF={rrf_score:.5f} | [{in_vec}{in_bm}] | Vec={vec_rank_str:>2s} BM25={bm25_rank_str:>2s} | {app:12s} | {fname}{mark}")
            if efile and meta.get("source_file") == efile and efile_rrf_rank is None:
                efile_rrf_rank = rank

        if efile and efile_rrf_rank is None:
            for rank, (cid, rrf_score) in enumerate(sorted_rrf):
                if id_to_meta.get(cid, {}).get("source_file") == efile:
                    efile_rrf_rank = rank
                    print(f"\n  [!] Fichier attendu au rang RRF {rank} (hors top {top_n}) | score={rrf_score:.5f}")
                    break

        if efile:
            if efile_rrf_rank is not None:
                steps_results.append(("RRF", "PASS", f"Rang RRF = {efile_rrf_rank}"))
            else:
                steps_results.append(("RRF", "FAIL", "Fichier absent des candidats RRF"))

        # Chunks au-dela de top_k
        rejected_by_topk = sorted_rrf[TOP_K:]
        if rejected_by_topk:
            print(f"\n  Chunks au-dela de top_k={TOP_K} (exclus du prompt) : {len(rejected_by_topk)}")
            for cid, score in rejected_by_topk[:5]:
                meta  = id_to_meta.get(cid, {})
                fname = meta.get("source_file", "?")[:55]
                mark  = " <-- FICHIER ATTENDU" if efile and meta.get("source_file") == efile else ""
                print(f"    RRF={score:.5f} | {fname}{mark}")

        # ── Etape 8 : Selection top_k ─────────────────────────────────────────
        print(f"\n[08] SELECTION TOP_K = {TOP_K}")
        sep()
        top_k_ids = set(cid for cid, _ in sorted_rrf[:TOP_K])

        if efile and efile_rrf_rank is not None:
            if efile_rrf_rank < TOP_K:
                report_step(8, "SELECTION TOP_K", "PASS",
                    proof=f"Rang RRF {efile_rrf_rank} < {TOP_K} -> inclus dans le prompt")
                steps_results.append(("TOP_K", "PASS", ""))
            else:
                report_step(8, "SELECTION TOP_K", "FAIL",
                    cause=f"Rang RRF {efile_rrf_rank} >= top_k ({TOP_K})",
                    recom="Le fichier est retrouve mais classe trop bas. Un autre document domine le corpus.")
                steps_results.append(("TOP_K", "FAIL", f"Rang {efile_rrf_rank} >= {TOP_K}"))
                stop_trace = True
        elif efile and efile_rrf_rank is None:
            report_step(8, "SELECTION TOP_K", "FAIL",
                cause="Fichier absent des candidats RRF -> absent du prompt",
                recom="Fichier non retourne ni par BM25 ni par vectoriel")
            steps_results.append(("TOP_K", "FAIL", "Absent des candidats RRF"))
            stop_trace = True
        else:
            top_k_selected_info = sorted_rrf[:TOP_K]
            docs_sel = [id_to_meta.get(cid, {}).get("source_file", "?") for cid, _ in top_k_selected_info]
            unique_f = list(dict.fromkeys(docs_sel))
            print(f"  {len(top_k_selected_info)} chunks selectionnes ({len(unique_f)} fichiers sources) :")
            for f in unique_f:
                cnt = docs_sel.count(f)
                print(f"    {cnt:2d} chunk(s) | {f}")

        # ── TABLEAU RECAPITULATIF PAR DOCUMENT ────────────────────────────────
        build_document_table(
            bm25_all_positive, vec_ids, vec_metas, sorted_rrf,
            top_k_ids, id_to_meta, chroma_metas, efile
        )

        # ── RECHERCHE EXHAUSTIVE (si --exhaustive) ────────────────────────────
        if args.exhaustive and qv is not None:
            exhaustive_search(
                qv, bscores, sidx, collection, chroma_metas, chroma_ids,
                idx_to_id, id_to_meta, chroma_filter, efile, total_chunks
            )

    # ─── ETAPE 9 : CONSTRUCTION DU PROMPT ────────────────────────────────────
    if not stop_trace:
        header("ETAPE 9 — Construction du prompt")

        top_k_selected = sorted_rrf[:TOP_K]
        relaxed_used   = False

        if len(top_k_selected) < 3 and chroma_filter is not None:
            relaxed_used = True
            print("  [!] Repli elargi active (< 3 resultats avec filtre actif)")
            print("      Recherche sans filtre...")
            try:
                vr2 = collection.query(
                    query_embeddings=[qv],
                    n_results=min(N_RESULTS, total_chunks),
                    include=["documents", "metadatas"]
                )
                vec_ids2 = vr2.get("ids", [[]])[0]
            except Exception:
                vec_ids2 = []

            sidx2_ids = [idx_to_id[i] for i in sidx if bscores[i] > 0][:N_RESULTS]
            rrf2 = {}
            for rank, cid in enumerate(vec_ids2):
                rrf2[cid] = rrf2.get(cid, 0) + 1 / (rank + 1 + RRF_K)
            for rank, cid in enumerate(sidx2_ids):
                rrf2[cid] = rrf2.get(cid, 0) + 1 / (rank + 1 + RRF_K)
            sorted_rrf2    = sorted(rrf2.items(), key=lambda x: x[1], reverse=True)
            top_k_selected = sorted_rrf2[:TOP_K]

        if efile:
            efile_in_prompt = [cid for cid, _ in top_k_selected
                               if id_to_meta.get(cid, {}).get("source_file") == efile]
            if efile_in_prompt:
                report_step(9, "CONSTRUCTION DU PROMPT", "PASS",
                    proof=f"{len(efile_in_prompt)} chunk(s) du fichier inclus | relaxed={relaxed_used}")
                steps_results.append(("PROMPT", "PASS", ""))
            else:
                report_step(9, "CONSTRUCTION DU PROMPT", "FAIL",
                    cause="Fichier absent du prompt final (evince meme apres repli elargi)",
                    recom="Ce fichier est domine par d'autres documents meme sans filtre actif")
                steps_results.append(("PROMPT", "FAIL", "Absent du prompt final"))
                stop_trace = True
        else:
            docs_in_prompt = [id_to_meta.get(cid, {}).get("source_file", "?") for cid, _ in top_k_selected]
            unique_prompt  = list(dict.fromkeys(docs_in_prompt))
            print(f"  {len(top_k_selected)} chunks dans le prompt | relaxed={relaxed_used}")
            print(f"  {len(unique_prompt)} fichiers sources uniques :")
            for f in unique_prompt:
                cnt = docs_in_prompt.count(f)
                app = id_to_meta.get(next((cid for cid, _ in top_k_selected
                                          if id_to_meta.get(cid, {}).get("source_file") == f), ""), {}).get("application", "?")
                print(f"    {cnt:2d} chunk(s) | {app:12s} | {f}")
            steps_results.append(("PROMPT", "PASS", f"{len(unique_prompt)} fichiers sources"))

        if args.show_chunks > 0:
            print(f"\n  Apercu des chunks envoyes au LLM (--show-chunks {args.show_chunks}) :")
            for rank, (cid, _) in enumerate(top_k_selected):
                meta = id_to_meta.get(cid, {})
                fname = meta.get("source_file", "?")
                doc = id_to_doc.get(cid, "")
                snippet = doc[:args.show_chunks].replace('\n', ' ')
                suffix = "..." if len(doc) > args.show_chunks else ""
                print(f"    [Chunk {rank:2d}] {fname[:30]:30s} | {len(doc):4d} chars | {snippet}{suffix}")

        # ─── ETAPES 10-11 : LLM ──────────────────────────────────────────────
        ans = None
        if args.run_llm and not stop_trace:
            header("ETAPES 10-11 — Generation LLM (Ollama)")
            print("  Appel Ollama (qwen3:8b)...")

            context_str = ""
            for i, (cid, _) in enumerate(top_k_selected):
                doc = id_to_doc.get(cid, "")
                context_str += f"[SOURCE {i}]\n{doc}\n---\n"

            prompt_text = f"Contexte:\n{context_str}\n\nQuestion: {q}"
            try:
                import ollama
                res = ollama.chat(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": prompt_text}],
                    options={"temperature": 0.2}
                )
                ans = res["message"]["content"]
                print(f"  Reponse generee ({len(ans)} chars)")
            except Exception as e:
                print(f"  [ERREUR] Impossible d'appeler Ollama : {e}")

    # ─── CONCLUSION ──────────────────────────────────────────────────────────
    if args.investigation:
        category = generate_forensic_report(q, qv, collection, chroma_filter, id_to_meta, id_to_doc,
                                 bm25_all_positive, vec_ids, vec_distances, sorted_rrf, top_k_selected,
                                 ans, N_RESULTS, TOP_K)
        print()
        return category

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION BATCH (PHASE 7)
# ─────────────────────────────────────────────────────────────────────────────
def run_validation_suite(args, collection, embedding_model, bm25, chroma_metas, chroma_ids, chroma_docs, id_to_meta, id_to_doc, idx_to_id, total_chunks):
    print("\n=========================================================")
    print("      LANCEMENT DE LA VALIDATION SUITE (50 Tests)")
    print("=========================================================\n")
    print("Exécution en cours. Le processus principal sera silencieux...")
    
    import sys, os
    
    results = {
        "CORPUS VIDE": 0,
        "FILTRE BLOQUANT": 0,
        "BRUIT BM25": 0,
        "BRUIT VECTORIEL": 0,
        "BRUIT RRF": 0,
        "CHUNK INCOMPLET": 0,
        "HALLUCINATION LLM": 0,
        "REJET LLM (Capacité insuffisante)": 0,
        "NON JUGEABLE": 0,
        "ERREUR INTERNE": 0
    }
    
    total = len(VALIDATION_SUITE)
    success_count = 0
    
    for i, test_query in enumerate(VALIDATION_SUITE):
        print(f"\rProgression : [{i+1}/{total}] '{test_query[:40]}...' ", end="", flush=True)
        
        # Rediriger stdout pour eviter le bruit
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            cat = run_single_trace(test_query, args, collection, embedding_model, bm25, chroma_metas, chroma_ids, chroma_docs, id_to_meta, id_to_doc, idx_to_id, total_chunks)
            if cat in ["HALLUCINATION LLM", "REJET LLM (Capacité insuffisante)"]:
                # Le chunk a atteint le LLM, c'est un succes du point de vue Retrieval
                success_count += 1
            if cat:
                results[cat] = results.get(cat, 0) + 1
        except Exception:
            results["ERREUR INTERNE"] += 1
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
    print("\n\n=========================================================")
    print("      RAPPORT DE VALIDATION RAG (50 Questions)")
    print("=========================================================")
    print(f"Total exécuté      : {total}")
    print(f"Réussites Retrieval: {success_count} ({success_count/total*100:.1f}%) -> Information a atteint le LLM")
    print(f"Échecs Retrieval   : {total - success_count} ({(total - success_count)/total*100:.1f}%) -> Perdue en cours de route\n")
    
    print("--- RÉPARTITION DES CAUSES SCIENTIFIQUES ---")
    # Tri par fréquence
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for cause, count in sorted_res:
        if count > 0:
            print(f"{cause:<35}: {count:2d} ({count/total*100:.1f}%)")
    print("=========================================================\n")

def main():
    parser = argparse.ArgumentParser(
        description="Outil de diagnostic universel du pipeline RAG CorporateBrain v3.0"
    )
    parser.add_argument("--query",         type=str, default=None,
                        help="Question utilisateur a diagnostiquer (requis hors mode validation)")
    parser.add_argument("--validation-suite", action="store_true",
                        help="Executer le batch complet de 50 questions de validation")
    parser.add_argument("--expected-file", type=str, default=None,
                        help="(Optionnel) Nom du fichier censé contenir la reponse")
    parser.add_argument("--filter",        type=str, default=None,
                        help='Filtre JSON (ex: \'{"application": "KPSA"}\')')
    parser.add_argument("--top-n",         type=int, default=15,
                        help="Nombre de resultats a afficher dans les classements (defaut: 15)")
    parser.add_argument("--exhaustive",    action="store_true",
                        help="Recherche le rang reel dans TOUT le corpus (pas seulement top 10)")
    parser.add_argument("--search-text",   type=str, default=None,
                        help="Effectuer une recherche plein texte independante du retrieval")
    parser.add_argument("--show-chunks",   type=int, default=0,
                        help="Afficher les N premiers caracteres de chaque chunk du prompt")
    parser.add_argument("--investigation", action="store_true",
                        help="Mode investigation : diagnostic complet et autopsie automatique")
    parser.add_argument("--run-llm",       action="store_true",
                        help="Executer les etapes 10-11 avec Ollama (plus lent)")
    args = parser.parse_args()
    
    if not args.query and not args.validation_suite:
        parser.error("L'argument --query est requis, a moins d'utiliser --validation-suite.")

    if args.validation_suite:
        args.investigation = True
        args.run_llm = True
    elif args.investigation:
        args.exhaustive = True
        args.run_llm = True

    header("CHARGEMENT DES BACKENDS")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    total_chunks = collection.count()
    if total_chunks == 0:
        print("  ERREUR : ChromaDB est vide. Aucun document ingere.")
        return

    embedding_model = SentenceTransformer(EMBED_MODEL)

    all_data     = collection.get(include=["documents", "metadatas"])
    chroma_docs  = all_data["documents"]
    chroma_metas = all_data["metadatas"]
    chroma_ids   = all_data["ids"]

    id_to_meta = {cid: meta for cid, meta in zip(chroma_ids, chroma_metas)}
    id_to_doc  = {cid: doc  for cid, doc  in zip(chroma_ids, chroma_docs)}
    idx_to_id  = {i: cid    for i, cid    in enumerate(chroma_ids)}

    bm25 = BM25Okapi([d.lower().split() for d in chroma_docs])

    print(f"  ChromaDB : {total_chunks} chunks indexes")
    print(f"  BM25     : {total_chunks} documents indexes")
    print(f"  Modele   : {EMBED_MODEL}")
    
    if args.validation_suite:
        run_validation_suite(args, collection, embedding_model, bm25, chroma_metas, chroma_ids, chroma_docs, id_to_meta, id_to_doc, idx_to_id, total_chunks)
    else:
        run_single_trace(args.query, args, collection, embedding_model, bm25, chroma_metas, chroma_ids, chroma_docs, id_to_meta, id_to_doc, idx_to_id, total_chunks)

if __name__ == "__main__":
    main()
