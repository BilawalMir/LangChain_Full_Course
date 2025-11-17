from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)

model = ChatHuggingFace(llm=llm)


class review(BaseModel):
    key_themes: list[str] = Field(description="The main themes discussed in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal[
        Literal[
            "Positive",
            "Negative",
            "Mixed",
        ]
    ] = Field(description="The overall sentiment of the review")
    pros: Optional[list[str]] = Field(
        default=None, description="List of positive aspects mentioned in the review  "
    )
    cons: Optional[list[str]] = Field(
        default=None, description="List of negative aspects mentioned in the review"
    )
    name: Optional[str] = Field(default=None, description="Name of the reviewer")


structured_output = model.with_structured_output(
    review
)  # this will show error because TinyLlama model does not support structured output

result = structured_output.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""
)

print(
    result
)  # {'summary': 'The hardware is great, but the software has many bugs and crashes often. The UI is outdated and not user-friendly.', 'sentiment': 'Mixed'}ng import TypedDict
print(result)
print(result.summary)
print(result.name)
