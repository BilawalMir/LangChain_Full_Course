from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
)

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about  {topic}", input_variable=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a linkdin post about - {topic}", input_variable=["topic"]
)

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkdin_post": RunnableSequence(prompt2, model, parser),
    }
)

result = parallel_chain.invoke({"topic": "AI"})

print(result["tweet"])
print(result["linkdin_post"])
