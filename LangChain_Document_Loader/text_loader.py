from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt = PromptTemplate(
    template= 'write the summary of the following poem - \n {poem}'
    input_variables=['poem']
)

loader = TextLoader("D:\LangChain_Models\LangChain_Document_Loader\cricket.txt", encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

chain.invoke({'poem': docs[0].page_content})