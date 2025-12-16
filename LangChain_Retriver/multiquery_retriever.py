from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.retrivers.multi_query import MultiQueryRetriever

load_dotenv()

embeddings = OpenAIEmbeddings()

all_docs = [
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

vector_store = FAISS.from_documents(
    documents=all_docs,
    embeddings=embeddings,
)
# Simple Retriever
similarity_retriever = vector_store.similarity_search(
    search_type="similarity", search_kwargs={"k": 2}
)

# MultiQuery Retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
)

query = "how to improve bowling attack "

similarity_retriever_results = similarity_retriever.invoke(query)
multiquery_retriever_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_retriever_results):
    print(f"\n --- {i+1}---")
    print(doc.page_content)


for i, doc in enumerate(multiquery_retriever_results):
    print(f"\n --- {i+1}---")
    print(doc.page_content)
