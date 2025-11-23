from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="D:/LangChain_Models/LangChain_Document_Loader/books",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)
docs = loader.load()

print(len(docs))

print(docs[325].page_content)
print(docs[325].metadata)

#OR
#OR
docs1 = loader.lazy_load()

for document in docs1:
    print(document.metadata)
