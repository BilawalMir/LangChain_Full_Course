from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate(
    [
        ("system", "You are an expert in {domain}."),
        ("human", "Explain the topic of {topic} in detail."),
    ]
)

prompt = chat_template.invoke({"domain": "physics", "topic": "quantum computing"})

print(prompt)
