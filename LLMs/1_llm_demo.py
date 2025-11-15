from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

result = llm.invoke("Tell me a joke about programming.")

print(result)
# O --- IGNORE --- Example output: "Why do programmers prefer dark mode? Because light attracts bugs!"