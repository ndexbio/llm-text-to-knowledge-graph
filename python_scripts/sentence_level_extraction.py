import json
from get_interactions import initialize_chains
import time
from grounding_genes import ground_genes


def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


# Initialize dictionaries to store the results
llm_results = {}


#extracting bel functions using llm
def llm_bel_processing(paragraphs, api_key):
    bel_extraction_chain = initialize_chains(api_key)
    
    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    # Loop through the sentences dictionary directly
    for index, paragraph_info in paragraphs.items():
        paragraph = paragraph_info['text']  # Assuming 'text' is the correct key for the sentence
        results = bel_extraction_chain.invoke({
            "text": paragraph
        })

        llm_results["LLM_extractions"].append({
            "Index": index,
            "text": paragraph,
            "Results": results
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
    return llm_results


#function to create sub-interaction type-obj
def combine_interaction(interaction):
    if 'subject' in interaction and 'object' in interaction:
        return f"{interaction['subject']} {interaction['interaction_type']} {interaction['object']}"
    elif 'enz' in interaction and 'sub' in interaction:
        return f"{interaction['enz']['name']} {interaction['type']} {interaction['sub']['name']}"
    elif 'subj' in interaction and 'obj' in interaction:
        return f"{interaction['subj']['name']} {interaction['type']} {interaction['obj']['name']}"
    elif 'members' in interaction and len(interaction['members']) == 2:
        member1, member2 = interaction['members']
        return f"{member1['name']} {interaction['type']} {member2['name']}"
    else:  # Handle unexpected formats
        return None, None, None


# Create combined results
def create_combined_results(results):
    combined_results = []
    for result in results:
        combined_entry = {
            "Index": result["Index"],
            "Sentence": result["Sentence"],
            "Combined_Results": list(set([combine_interaction(interaction) for interaction in result["Results"]]))
        }
        combined_results.append(combined_entry)
    return combined_results


# Combine LLM and small corpus extractions
def combine_llm_and_bel_extractions(llm_extractions, small_corpus_extractions):
    # Initialize the combined output dictionary
    combined_results = {}

    # Iterate through both LLM extractions and small corpus extractions
    for llm_extraction in llm_extractions["LLM_extractions"]:
        index = llm_extraction["Index"]
        text = llm_extraction["text"]

        # Extract LLM results
        llm_bel_statements = [item["bel_statement"] for item in llm_extraction["Results"]]

        # Extract small corpus results
        small_corpus_bel_statements = small_corpus_extractions.get(index, {}).get("bel_statements", [])

        # Combine into a single output format
        combined_results[index] = {
            "text": text,
            "LLM_bel_statements": llm_bel_statements,
            "Small_Corpus_bel_statements": small_corpus_bel_statements
        }
    return combined_results
