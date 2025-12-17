from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser

# STEP 1a INDEXING --------->
video_id = "Gfr50f6ZBvo"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, language=["en"])

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No caption available for this video")


# STEP 1b INDEXING --------->
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.add_documents(chunks, embeddings)

# Step 2 - Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

retriever.invoke("what is deepmind")

# Step 3 - Augmentation

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
        
       {context}
       question :{question}   
      """,
    input_variables=["context", "question"],
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

## Step 4 - Generation
answer = llm.invoke(final_prompt)
print(answer.content)


# BUILDING A CHAIN
def formate_doc(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(formate_doc),
        "question": RunnablePassthrough(),
    }
)

parallel_chain.invoke("who is devis")

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke("Can you summarize the video")
