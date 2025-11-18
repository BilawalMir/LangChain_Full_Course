from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="give me the name and age  and city of fictionanl person /n {format_instructions}",
    input_variables=[],
    partial_variables={"formate_instruction": parser.get_formate_instructions()},
)


# Chains Method

chain = template1 | model | parser

result = chain.invoke({})
print(result)

# Without Chains Method
prompt = template1.formate()

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
print(type(final_result))
print(final_result["name"])
 