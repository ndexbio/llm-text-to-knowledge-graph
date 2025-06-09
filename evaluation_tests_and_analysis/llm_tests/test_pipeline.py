import os
import argparse
import json
import warnings
import time
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from main_scripts.get_interactions import load_prompt
from main_scripts.grounding_genes import annotate_paragraphs_in_json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Rate limiting parameters
rate_limit_per_minute = 3
delay = 60.0 / rate_limit_per_minute


def delayed_completion(model_type: str, delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time and return appropriate model."""
    time.sleep(delay_in_seconds)

    if model_type.lower() == "openai":
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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


# Helper to simplify annotations
def simplify_annotations(annotations):
    """
    Given a list of annotation dictionaries, return a list of strings in the format:
    "db:entry_name"
    """
    return [f"{ann.get('db')}:{ann.get('entry_name')}" for ann in annotations]


# Function to create the appropriate OpenAI extraction model with structured output
def create_extraction_model():
    model = delayed_completion(
        model_type="openai",
        delay_in_seconds=delay,
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    return model.with_structured_output(BELInteractions)


# Create the extraction chain, with optional special instructions
def create_extraction_chain(prompt: str, use_special: bool = False):
    """Create an extraction chain using OpenAI with or without special instructions."""
    # Append special instructions if flag is set
    if use_special:
        prompt = (
            prompt + " special additional instructions: " 
            " consider the confidence values in the annotations list when forming statements "
            "and if the confidence is low, make your decision on what is the appropriate official name "
            "and use that for the statement."
        )

    model = create_extraction_model()
    ann_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "{text} | Annotations: {annotations}")
    ])

    def validate_response(response):
        # Ensure the output is BELInteractions
        if hasattr(response, 'interactions'):
            return response
        if isinstance(response, dict) and 'interactions' in response:
            return BELInteractions(**response)
        if isinstance(response, list):
            return BELInteractions(interactions=response)
        return BELInteractions(interactions=[])

    return ann_prompt | model | validate_response


# Process text and return JSON using OpenAI only
def process_text_to_json(input_text, annotations, prompt: str, use_special: bool):
    # Simplify annotations list
    simple_annotations = simplify_annotations(annotations) if isinstance(annotations, list) else annotations

    # Invoke extraction chain
    extraction_chain = create_extraction_chain(prompt, use_special=use_special)
    result = extraction_chain.invoke({
        "text": input_text,
        "annotations": simple_annotations
    })
    return json.loads(result.model_dump_json())


def llm_processing(paragraphs, prompt: str, use_special: bool):
    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    for index, paragraph_info in paragraphs.items():
        sentence = paragraph_info['text']
        annotations = paragraph_info.get('annotations', [])
        # Filter out unwanted annotations
        filtered_annotations = [ann for ann in annotations if ann.get("db") not in ("MESH", "MESHD")]

        results_dict = process_text_to_json(
            sentence,
            filtered_annotations,
            prompt,
            use_special
        )
        interactions = results_dict.get('interactions', [])

        llm_results["LLM_extractions"].append({
            "Index": index,
            "text": sentence,
            "Results": interactions
        })

    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)") 
    return llm_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEL extraction pipeline on input JSON.")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the input JSON file with sentences for testing"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="minimal_prompt_1.txt",
        help="Path to a custom prompt file (optional)."
    )
    parser.add_argument(
        "--use_special",
        action="store_true",
        help="If set, include special additional instructions in the prompt."
    )
    args = parser.parse_args()

    # Load base prompt
    prompt = load_prompt(prompt_file=args.prompt_file)

    # Load and annotate paragraphs
    with open(args.input_json) as f:
        paragraphs = json.load(f)
    annotated_paragraphs = annotate_paragraphs_in_json(paragraphs)

    # Process with OpenAI only, optionally using special instructions
    results = llm_processing(annotated_paragraphs, prompt, use_special=args.use_special)

    # Save output
    with open("llm_results_2.json", "w") as out:
        json.dump(results, out, indent=2)
    print("BEL extraction completed.")
