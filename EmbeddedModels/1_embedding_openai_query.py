from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", dimensions=32)

result = embedding.embed_query("What is the capital of France?")

print(str(result))
# O --- IGNORE --- Example output: "[0.01234, -0.05678, 0.02345, ..., 0.03456]"