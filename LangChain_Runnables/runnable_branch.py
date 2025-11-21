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


def word_count(text):
    return len(text.split())


model = ChatOpenAI()

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template="Write a detail report on  {topic}", input_variable=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}", input_variables=["text"]
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)
print(final_chain.invoke({"topic": "Russia vs Ukraine"}))

# OR
final_chain1 = report_gen_chain | branch_chain
print(final_chain1.invoke({"topic": "Russia vs Ukraine"}))
