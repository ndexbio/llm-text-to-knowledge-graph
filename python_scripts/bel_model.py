import os
import warnings
import time
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time.""" 
    time.sleep(delay_in_seconds)
    return ChatOpenAI(**kwargs)


rate_limit_per_minute = 3
delay = 60.0 / rate_limit_per_minute


# Updated model schema for BEL extraction
class BELInteraction(BaseModel):
    """BEL interaction extracted from the sentence."""
    bel_statement: str = Field(..., description="A BEL formatted statement representing an interaction.")
    evidence: str = Field(
        ..., 
        description="The exact sentence from which the interacting subject and object is taken from"
    )


class BELInteractions(BaseModel):
    """BEL interaction collection for a sentence."""
    interactions: List[BELInteraction]


# Convert the BELInteractions schema for the OpenAI function
bel_extraction_function = [
    convert_pydantic_to_openai_function(BELInteractions)
]

# Define model for extraction
model = delayed_completion(delay_in_seconds=delay, model="gpt-4o",
                           temperature=0, openai_api_key=OPENAI_API_KEY)


bel_extraction_model = model.bind(
    functions=bel_extraction_function,
    function_call={"name": "BELInteractions"}
)
