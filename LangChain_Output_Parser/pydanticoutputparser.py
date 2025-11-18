from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    PydanticOutputParser,
)
from pydantic import BaseModel, Field


load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="Name of Person ")
    age: int = Field(gt=18, description="Age of Person ")
    city: str = Field(description="Name of City the person belong to ")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name , age and city of a fictional {place} person. /n {format_instructions}",
    input_variables=["place"],
    partial_variables={"formate_instruction": parser.get_format_instructions()},
)

# Chains Method
chain = template | model | parser

result = chain.invoke({"place": "pakistani"})

print(result)

# Without Chains Method

prompt = template.invoke({"place": "pakistani"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
