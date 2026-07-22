"""
Script de conversion automatique .doc -> .docx
Utilise Microsoft Word via l'interface COM (pywin32)
"""
import os
import sys

try:
    import win32com.client
except ImportError:
    print("ERREUR : pywin32 non installe. Lancez : pip install pywin32")
    sys.exit(1)

STORAGE_DIR = "doc_storage_v2"

def convert_doc_to_docx(doc_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    try:
        abs_path = os.path.abspath(doc_path)
        docx_path = abs_path + "x"
        doc = word.Documents.Open(abs_path)
        doc.SaveAs(docx_path, FileFormat=16)
        doc.Close(False)
        print(f"OK Converti : {os.path.basename(doc_path)}")
        return True
    except Exception as e:
        print(f"ERREUR pour {os.path.basename(doc_path)} : {e}")
        return False
    finally:
        word.Quit()

def main():
    doc_files = [
        f for f in os.listdir(STORAGE_DIR)
        if f.lower().endswith(".doc") and not f.lower().endswith(".docx")
    ]
    if not doc_files:
        print("Aucun fichier .doc a convertir.")
        return
    print(f"{len(doc_files)} fichier(s) .doc trouves...\n")
    for filename in doc_files:
        doc_path = os.path.join(STORAGE_DIR, filename)
        if convert_doc_to_docx(doc_path):
            try:
                os.remove(doc_path)
                print(f"  Ancien fichier supprime : {filename}")
            except Exception as e:
                print(f"  Impossible de supprimer {filename} : {e}")
    print("\nTermine !")

if __name__ == "__main__":
    main()
