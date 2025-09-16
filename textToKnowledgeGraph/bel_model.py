import os
import warnings
import time
from typing import List, Optional
from pydantic import BaseModel, Field
#from dotenv import load_dotenv
#from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
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


# # Convert the BELInteractions schema for the OpenAI function
# bel_extraction_function = [
#     convert_to_openai_function(BELInteractions)
# ]


def _prepare_credentials(api_key: Optional[str]) -> None:
    """
    Configure credentials. If api_key is "none", load from .env in current directory.
    """
    logging.info(f"_prepare_credentials: api_key={'none' if api_key == 'none' else 'provided' if api_key else 'None'}")
    if api_key == "none":
        # Load from .env in current directory; ChatLiteLLM/LiteLLM reads OPENAI_API_KEY from the environment.
        #logging.info("Loading API key and model settings from .env in current directory")
        #load_dotenv(override=False)

        # Configure LiteLLM settings
        if os.environ.get("LITELLM_DEBUG", "0") in ("1", "true", "True", "TRUE"):
            litellm._turn_on_debug()

        # Set drop_params to True to handle unsupported parameters gracefully
        litellm.drop_params = True

        # Suppress function call parsing warnings in LiteLLM
        litellm.suppress_debug_info = True

    else:
        if not api_key:
            raise ValueError("api_key must be provided when it is not 'none'.")


def _delayed_model(delay_in_seconds: float = 1.0, **kwargs) -> ChatOpenAI:
    """Delay model construction to respect a per-minute rate limit, then return ChatLiteLLM."""
    time.sleep(delay_in_seconds)
    return ChatOpenAI(**kwargs)


def initialize_model(
    api_key: Optional[str] = None,
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Initialize a ChatOpenAI model with either explicit api_key or .env-based credentials."""
    _prepare_credentials(api_key=api_key)

    # If api_key is "none", use local endpoint with ChatOpenAI directly
    if api_key == "none":
        # For local endpoint, use "none" as model name (matching your working example)
        logging.info("bel_model.py: Using model from local endpoint")

        return _delayed_model(
            delay_in_seconds=delay,
            model="none",  # the litellm proxy endpoint sets the model, so this has no effect
            temperature=temperature,
            openai_api_base="http://localhost:4000",
            api_key="none"
        )
    else:
        # Use standard OpenAI endpoint with provided API key
        logging.info(f"bel_model.py: Using model {model_name} with OpenAI API")

        return _delayed_model(
            delay_in_seconds=delay,
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )


def get_bel_extraction_model(
    api_key: Optional[str] = None,
    *,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0
):
    """Return a LLM instance bound to the BEL extraction function schema."""

    litellm.drop_params = True
    model = initialize_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )

    logging.info("bel_model.py: Model initialized")

    return model.bind(response_format=BELInteractions)
    #return model.bind(response_format={"type": "json_object"}) # triggers "arguments"

    # # Use bind_tools instead of bind with functions for better compatibility
    # try:
    #     return model.bind_tools([BELInteractions], strict=True)
    # except AttributeError:
    #     # Fallback to older method if bind_tools is not available
    #     logging.warning("bind_tools not available, falling back to functions binding")
    #     return model.bind(
    #         functions=bel_extraction_function,
    #         function_call={"name": "BELInteractions"},
    #     )
