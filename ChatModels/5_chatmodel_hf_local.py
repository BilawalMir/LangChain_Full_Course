from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
import os

os.environ['HF-HOME']= 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
  model_id="Tiny-llama/Llama-2-7b-chat-hf",
  task="text-generation",
  model_kwargs={"temperature":0, "max_new_tokens":100}
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Explain the theory of relativity in simple terms.")
print(result.content)
# O --- IGNORE --- Example output: "The theory of relativity, developed by Albert Einstein, consists of two main parts: special relativity and general relativity. Special relativity explains how time and space are linked for objects moving at a constant speed in a straight line, showing that time can slow down and lengths can contract as you approach the speed of light. General relativity expands on this by describing how gravity is not just a force between masses but a curvature of space and time caused by mass and energy. In simple terms, massive objects like planets and stars bend the space around them, which affects the motion of other objects, leading to what we perceive as gravity."
