from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", dimensions=300)

documents = [
    "Virat Kohli is a famous Indian cricketer.",
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science."
    "MS Dhoni is a former captain of the Indian cricket team."
    "Sachin Tendulkar is known as the 'king of Cricket'.",
]

query = "Who is Virat Kohli?"

doc_vectors = embedding.embed_documents(documents)
query_vector = embedding.embed_query(query)
scores = cosine_similarity([query_vector], doc_vectors)[0]

index, scores = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity scores :", scores)

# O --- IGNORE --- Example output:
# Who is Virat Kohli?
# Virat Kohli is a famous Indian cricketer.
# Similarity scores : 0.8723456789123456
