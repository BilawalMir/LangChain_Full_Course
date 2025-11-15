from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(temperature=0, model_name="gpt-4", max_completion_tokens=100)

result = model.invoke("Explain the theory of relativity in simple terms.")
print(result)
# O --- IGNORE --- Example output: "The theory of relativity, developed by Albert Einstein, consists of two main parts: special relativity and general relativity. Special relativity explains how time and space are linked for objects moving at a constant speed in a straight line, showing that time can slow down and lengths can contract as you approach the speed of light. General relativity expands on this by describing how gravity is not just a force between masses but a curvature of space and time caused by mass and energy. In simple terms, massive objects like planets and stars bend the space around them, which affects the motion of other objects, leading to what we perceive as gravity."
print(result.content)
