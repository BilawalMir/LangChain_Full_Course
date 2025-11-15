from langchain_core.messages import SystemMessage , HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
  SystemMessage(content = "You are a helpful research assistant."),
  HumanMessage(content = "Explain the theory of relativity in simple terms.")
]
result = model.invoke(messages)
messages.append(AIMessage(content = result.content))
print(messages)