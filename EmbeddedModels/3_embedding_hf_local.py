from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "The capital of France is Paris."

vector = embeddings.embed_query(text)
print(str(vector))
# O --- IGNORE --- Example output: "[-0.01234, 0.05678, -0.02345, ..., 0.03456]"  

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.", 
    "The capital of Italy is Rome."
]
vectors = embeddings.embed_documents(documents)
print(str(vectors))
# O --- IGNORE --- Example output: "[[-0.01234, 0.05678, -0.02345, ..., 0.03456], [0.01234, -0.04567, 0.03456, ..., -0.02345], [0.02345, -0.03456, 0.04567, ..., -0.01234]]" 

# Note: The actual numerical values in the output will vary based on the model's embeddings.
