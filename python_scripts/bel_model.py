import warnings
import time
from typing import List
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI
warnings.filterwarnings("ignore")


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


def initialize_model(api_key):
    """Initialize the model with the provided API key."""
    model = delayed_completion(
        delay_in_seconds=delay, 
        model="gpt-4o-mini",
        temperature=0, 
        openai_api_key=api_key
    )
    return model


# Ensure bel_extraction_model is initialized with the provided key
def get_bel_extraction_model(api_key):
    return initialize_model(api_key).bind(
        functions=bel_extraction_function,
        function_call={"name": "BELInteractions"}
    )
