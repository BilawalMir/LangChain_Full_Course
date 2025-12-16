from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

documents = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons."
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure."
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary."
    ),
]

vector_store = Chroma.from_documents(
    documents=documents,
    embeddings=embeddings,
    collection_name="my_collection",
)

query = "Who is the most successful captain in IPL history?"

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- results {i+1} ----")
    print(doc.page_content)
