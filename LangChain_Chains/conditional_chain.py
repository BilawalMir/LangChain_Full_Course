from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="give The sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
  template="Classify the sentiment of the following review as Positive, Negative or Neutral \n feedback: {feedback} \n {format_instructions}",
  input_variables=['feedback']
  partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2 

prompt2 = PromptTemplate(
  template=" generate an appropriate response to the positive feedback \n  {feedback}",
  input_variables=['feedback']
)

prompt3 = PromptTemplate(
  template=" generate an appropriate response to the negative feedback \n  {feedback}",
  input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model |parser),
    (lambda x:x.sentiment == "negative" , prompt3 | model | parser),
    RunnableLambda(lambda x: "could not classify the sentiment properly")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The product quality is excellent and I am very satisfied with my purchase!"})

print(result)
chain.get_graph().print_ascii()  # To visualize the chain structure