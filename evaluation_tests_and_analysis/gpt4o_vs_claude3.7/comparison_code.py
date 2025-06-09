import os
import json
import warnings
import time
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from main_scripts.get_interactions import prompt
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# Define a function that adds a delay to API calls
def delayed_completion(model_type: str, delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time and return appropriate model.""" 
    time.sleep(delay_in_seconds)

    if model_type.lower() == "openai":
        return ChatOpenAI(**kwargs)
    elif model_type.lower() == "anthropic":
        return ChatAnthropic(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Rate limiting parameters
rate_limit_per_minute = 3
delay = 60.0 / rate_limit_per_minute


# Pydantic schema definition for BEL extraction
class BELInteraction(BaseModel):
    """BEL interaction extracted from the sentence."""
    bel_statement: str = Field(..., description="A BEL formatted statement representing an interaction.")
    evidence: str = Field(
        ..., 
        description="The exact sentence from which the interacting subject and object is taken from"
    )


class BELInteractions(BaseModel):
    """BEL interaction collection for a sentence."""
    interactions: List[BELInteraction] = Field(default_factory=list)


def simplify_annotations(annotations):
    """
    Given a list of annotation dictionaries, return a list of strings in the format:
    "db:entry_name"
    """
    return [f"{ann.get('db')}:{ann.get('entry_name')}" for ann in annotations]


# Add this helper function near the top of your file
def parse_claude_response(raw_response):
    """Parse Claude's raw JSON response into interactions list."""
    try:
        # Check if response appears to be truncated
        if raw_response.count('"bel_statement"') > raw_response.count('"evidence"'):
            print("Detected truncated JSON response, attempting repair")

            # Find the last complete interaction
            last_complete = raw_response.rfind('    }')
            if last_complete > 0:
                # Truncate to the last complete interaction and close the JSON structure
                fixed_response = raw_response[:last_complete + 5] + "\n  ]\n}"
                parsed = json.loads(fixed_response)
            else:
                # Can't find a good truncation point, handle as best we can
                return []

        # Regular parsing logic
        elif "```json" in raw_response:
            json_content = raw_response.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(json_content)
        else:
            # Try to parse directly
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                # If fails, try to fix common format issues
                fixed_response = raw_response.replace('[\n  "interactions"', '[\n  {"interactions"')
                fixed_response = fixed_response.replace(']\n]', '}\n]')
                parsed = json.loads(fixed_response)

        # Convert to proper structure
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], dict) and "interactions" in parsed[0]:
                return parsed[0].get("interactions", [])
            elif "interactions" in parsed:
                # Handle case where it's an array with "interactions" as a direct key (invalid JSON)
                return parsed.get("interactions", [])
        elif isinstance(parsed, dict) and "interactions" in parsed:
            return parsed.get("interactions", [])

        return []
    except Exception as e:
        print(f"Error parsing Claude response: {e}")

        # Emergency fallback: manually extract BEL statements using regex
        try:
            import re
            pattern = r'"bel_statement":\s*"([^"]+)",\s*"evidence":\s*"([^"]+)"'
            matches = re.findall(pattern, raw_response)

            interactions = []
            for bel, evidence in matches:
                interactions.append({
                    "bel_statement": bel,
                    "evidence": evidence
                })

            if interactions:
                print(f"Recovered {len(interactions)} interactions using regex fallback")
                return interactions
        except Exception as regex_error:
            print(f"Regex fallback also failed: {regex_error}")

        return []


# Function to create the appropriate model with structured output
def create_extraction_model(model_provider: str = "openai"):
    """Create a model based on the specified provider with structured output."""
    if model_provider.lower() == "openai":
        model = delayed_completion(
            model_type="openai",
            delay_in_seconds=delay,
            model="gpt-4o",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        # Apply structured output to OpenAI model
        return model.with_structured_output(BELInteractions)

    elif model_provider.lower() == "anthropic":
        # Just return the basic model for Anthropic
        return delayed_completion(
            model_type="anthropic",
            delay_in_seconds=delay,
            model="claude-3-7-sonnet-latest",
            temperature=0,
            anthropic_api_key=ANTHROPIC_API_KEY
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


# Function to create the extraction chain
def create_extraction_chain(model_provider=None):
    """Create an extraction chain using the specified model provider."""
    if model_provider is None:
        raise ValueError("model_provider must be specified ('openai' or 'anthropic')")

    # For OpenAI, use the standard approach
    if model_provider.lower() == "openai":
        model = create_extraction_model(model_provider)
        ann_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{text} | Annotations: {annotations}")
        ])

        # Add a response validator step
        def validate_response(response):
            # Handle various response formats
            if isinstance(response, dict):
                if 'interactions' not in response:
                    return BELInteractions(interactions=[])
                return response
            elif isinstance(response, list):
                return BELInteractions(interactions=response)
            elif not hasattr(response, 'interactions'):
                return BELInteractions(interactions=[])
            return response

        return ann_prompt | model | validate_response

    # For Anthropic, we don't need a chain since we handle it directly in process_text_to_json
    else:
        return None


# Function to process text and return JSON
def process_text_to_json(input_text, annotations="", model_provider=None):
    if model_provider is None:
        raise ValueError("model_provider must be specified ('openai' or 'anthropic')")

    # Simplify annotations
    if isinstance(annotations, list):
        simple_annotations = simplify_annotations(annotations)
    else:
        simple_annotations = annotations

    # Format the prompt
    ann_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "{text} | Annotations: {annotations}")
    ])
    formatted_prompt = ann_prompt.format(
        text=input_text, 
        annotations=simple_annotations
    )

    if model_provider.lower() == "anthropic":
        # Direct approach for Anthropic models
        model = create_extraction_model(model_provider)
        raw_response = model.predict(formatted_prompt)
        print(f"Raw Claude response (first 500 chars): {raw_response[:500]}...")

        # Parse the JSON response using our helper function
        interactions = parse_claude_response(raw_response)

        # Create the result dictionary
        result_dict = {
            "interactions": interactions,
            "raw_response": raw_response
        }
    else:
        # Use the extraction chain for OpenAI
        extraction_chain = create_extraction_chain(model_provider)
        result = extraction_chain.invoke({
            "text": input_text,
            "annotations": simple_annotations
        })
        result_dict = json.loads(result.model_dump_json())

    return json.dumps(result_dict, indent=2)


def llm_processing(paragraphs, model_provider=None):
    if model_provider is None:
        raise ValueError("model_provider must be specified ('openai' or 'anthropic')")

    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    for index, paragraph_info in paragraphs.items():
        sentence = paragraph_info['text']
        annotations = paragraph_info.get('annotations', [])
        # Filter out MESH / MESHD annotations as before
        filtered_annotations = [ann for ann in annotations if ann.get("db") not in ("MESH", "MESHD")]

        # Use process_text_to_json to get JSON string
        json_result = process_text_to_json(sentence, filtered_annotations, model_provider)
        results_dict = json.loads(json_result)
        interactions = results_dict.get('interactions', [])
        raw_response = results_dict.get('raw_response', None)
        error = results_dict.get('error', None)

        # If we have a raw response but no interactions, try to extract from raw response again
        if raw_response and not interactions and model_provider.lower() == "anthropic":
            interactions = parse_claude_response(raw_response)

        result_entry = {
            "Index": index,
            "text": sentence,
            "Results": interactions,
            "error": error,
            "raw_response": raw_response
        }
        llm_results["LLM_extractions"].append(result_entry)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
    return llm_results
