from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
)

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variable=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following Joke - {text}", input_variable=["text"]
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "Explanation": RunnableSequence(prompt2, model, parser),
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic": "AI"})

print(result)
