import os
import json
import time
import logging
from .get_interactions import build_bel_extraction_chain, load_prompt
from functools import lru_cache
from importlib import resources

logger = logging.getLogger(__name__)


def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


@lru_cache(maxsize=6)
def _build_chain_cached(prompt_file, prompt_identifier, api_key, prompt_mtime):
    """
    Real chain builder wrapped by lru_cache.
    `prompt_mtime` is only used to make the cache key unique when the
    file changes; we never use its value inside the function body.
    """
    prompt_txt = load_prompt(prompt_file, prompt_identifier)
    return build_bel_extraction_chain(prompt_txt, api_key)


def _build_chain(prompt_file, prompt_identifier, api_key):
    # locate the file that lives in textToKnowledgeGraph/…
    data = resources.files("textToKnowledgeGraph").joinpath(prompt_file)

    # convert to a real on-disk path (works for wheels/zip-imports)
    with resources.as_file(data) as abs_path:
        mtime = os.path.getmtime(abs_path)
        return _build_chain_cached(abs_path, prompt_identifier, api_key, mtime)


# Initialize dictionaries to store the results
llm_results = {}


#Extracting BEL interactions from sentences with annotations using llm
def llm_bel_processing(paragraphs, api_key, 
                       prompt_file="prompt_file_v7.txt",
                       prompt_identifier="general prompt"):
    bel_extraction_chain = _build_chain(prompt_file, prompt_identifier, api_key)
    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    # Loop through the sentences dictionary directly
    for index, paragraph_info in paragraphs.items():
        sentence = paragraph_info['text']
        annotations = paragraph_info.get('annotations', [])  # Default to empty list if no annotations
        # Clean annotations to ensure they have 'db' and 'entry_name'
        clean_annotations = [
            {"db": ann["db"], "entry_name": ann["entry_name"]}
            for ann in annotations
            if "db" in ann and "entry_name" in ann
        ]
        # Invoke the BEL extraction chain with the sentence and cleaned annotations
        results = bel_extraction_chain.invoke({
            "text": sentence,
            "annotations": clean_annotations
        })

        llm_results["LLM_extractions"].append({
            "Index": index,
            "text": sentence,
            "Results": results,
            "annotations": annotations
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
    return llm_results
