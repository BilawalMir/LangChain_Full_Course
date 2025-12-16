from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.retrivers.contextual_compression  import ContextualCompressionRetriever
from langchain.retrievers.documents_compressors import LLMChainExtractor

load_dotenv()

embeddings = OpenAIEmbeddings()

docs = [
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
    documents=docs,
    embeddings=embeddings,
)

base_retriever = vector_store.as_retriever(search_kwargs = {"k": 5})

llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

#create contextual compressor retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    compressor=compressor,
)

query = "what is photosynthesis?"

compressed_results = compression_retriever.invoke(query)


for i , doc in enumerate(compressed_results):
    print(f"\n --- {i+1}---")
    print(doc.page_content)



 