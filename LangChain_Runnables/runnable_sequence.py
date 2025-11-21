from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnableSequence

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variable=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following Joke - {text}",
    input_variable=["text"]
)

chain = RunnableSequence(prompt , model , parser , prompt2 , model , parser)

result = chain.invoke({"topic": "AI"})

print(result)
