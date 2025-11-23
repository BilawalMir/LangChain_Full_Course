from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:\LangChain_Models\LangChain_Document_Loader\dl-curriculum.pdf")

docs = loader.load()

print(len(docs))

print(type(docs))

print(docs[0].page_content)
print(docs[1].metadata)
