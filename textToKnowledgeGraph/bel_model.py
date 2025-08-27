import os
import warnings
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_community.chat_models import ChatLiteLLM
import litellm
import logging

litellm.drop_params = True
warnings.filterwarnings("ignore")

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


def _prepare_credentials(api_key: Optional[str]) -> None:
    """
    Configure credentials for ChatLiteLLM.

    - If api_key is "none", load from .env in current directory.
    - Else, fall back to explicit api_key by setting OPENAI_API_KEY env var for this process.
    """
    logging.info(f"_prepare_credentials: api_key={'none' if api_key == 'none' else 'provided' if api_key else 'None'}")
    if api_key == "none":
        # Load from .env in current directory; ChatLiteLLM/LiteLLM reads OPENAI_API_KEY from the environment.
        logging.info("Loading API key and model settings from .env in current directory")
        load_dotenv(override=False)

        if os.environ.get("LITELLM_DEBUG", "0") in ("1", "true", "True", "TRUE"):
            litellm._turn_on_debug()
    else:
        if not api_key:
            raise ValueError("api_key must be provided when it is not 'none'.")
        logging.info("Setting OPENAI_API_KEY in environment from provided api_key")
        os.environ["OPENAI_API_KEY"] = api_key  # local process env only


def _delayed_model(delay_in_seconds: float = 1.0, **kwargs) -> ChatLiteLLM:
    """Delay model construction to respect a per-minute rate limit, then return ChatLiteLLM."""
    time.sleep(delay_in_seconds)
    return ChatLiteLLM(**kwargs)


def initialize_model(
    api_key: Optional[str] = None,
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> ChatLiteLLM:
    """Initialize a LiteLLM-backed chat model with either explicit api_key or .env-based credentials."""
    _prepare_credentials(api_key=api_key)

    # If api_key is "none", try to use MODEL_NAME from environment
    if api_key == "none" and "MODEL_NAME" in os.environ:
        model_name = os.environ["MODEL_NAME"]
        logging.info(f"Using model from environment: {model_name}")

    logging.info(f"bel_model.py: Using model {model_name} for extraction")

    return _delayed_model(
        delay_in_seconds=delay,
        model=model_name,
        temperature=temperature,
    )


def get_bel_extraction_model(
    api_key: Optional[str] = None,
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
):
    """Return a ChatLiteLLM instance bound to the BEL extraction function schema."""

    model = initialize_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )
    return model.bind(
        functions=bel_extraction_function,
        function_call={"name": "BELInteractions"},
    )
