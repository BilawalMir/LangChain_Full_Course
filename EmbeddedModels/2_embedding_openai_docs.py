from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", dimensions=32)

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.", 
    "The capital of Italy is Rome."
]

result = embedding.embed_documents(documents)

print(str(result))
# O --- IGNORE --- Example output: "[0.01234, -0.05678, 0.02345, ..., 0.03456]"