from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
)

load_dotenv()


def word_count(text):
    return len(text.split())


model = ChatOpenAI()

parser = StrOutputParser()


prompt = PromptTemplate(template="Write a joke about {topic}", input_variable=["topic"])

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_count)}
)

# OR
parallel_chain1 = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(lambda x: len(x.split())),
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic": "AI"})
final_result = """{} \n word count - {}""".formate(result["joke"], result["word_count"])
print(final_result)
