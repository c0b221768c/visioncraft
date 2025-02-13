import faiss

index = faiss.IndexFlatL2(512)
faiss.write_index(index, "/app/data/database.faiss")