from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Loaded!")

emb = model.encode("Hello world")
print(emb.shape)