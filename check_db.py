import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

print(f"Nombre de chunks : {collection.count()}")
print(collection.get())