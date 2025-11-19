from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name="claude-3-7-sonnet-20240922")


prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following  text \n {text}?",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate  5 short question answer from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
  template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz-> {quiz}"
  input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
  {
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
  }
 )

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction. AI applications include expert systems, natural language processing (NLP), speech recognition, and machine vision. The ultimate goal of AI is to create technology that allows computers and machines to function in an intelligent manner."""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()  # To visualize the chain structure
