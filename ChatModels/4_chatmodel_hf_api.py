from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = ChatHuggingFace(
  repo_id="Tiny-llama/Llama-2-7b-chat-hf",
  task="text-generation",
  temperature=0,
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("Explain the theory of relativity in simple terms.")
print(result)
# O --- IGNORE --- Example output: "The theory of relativity, developed by Albert
