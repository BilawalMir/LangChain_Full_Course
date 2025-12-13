from langchain.experimental.text_splitter import SematicChunker
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold="standard deviation",
    breakpoint_breakpoint_amount=1,
)

sample = """Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These intelligent machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."""
docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
