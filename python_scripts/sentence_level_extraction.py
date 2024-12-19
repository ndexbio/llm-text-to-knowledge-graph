import json
from get_interactions import extraction_chain, bel_extraction_chain
import time
from grounding_genes import ground_genes


def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


# Initialize dictionaries to store the results
llm_results = {}


#extracting interactions from sentences without annotations using llm
def llm_processing(sentences):
    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    # Loop through the sentences dictionary directly
    for index, sentence_info in sentences.items():
        sentence = sentence_info['text']  # Assuming 'text' is the correct key for the sentence
        results = extraction_chain.invoke({
            "text": sentence
        })
        llm_results["LLM_extractions"].append({
            "Index": index,
            "text": sentence,
            "Results": results
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
    return llm_results


#extracting bel functions using llm
def llm_bel_processing(paragraphs):
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


#extracting interactions from sentences with annotations using llm
def llm_ann_processing(sentences):
    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    # Loop through the sentences dictionary directly
    for index, sentence_info in sentences.items():
        sentence = sentence_info['text']
        annotations = sentence_info.get('annotations', [])  # Default to empty list if no annotations
        results = extraction_chain.invoke({
            "text": sentence,
            "annotations": annotations
        })
        llm_results["LLM_extractions"].append({
            "Index": index,
            "text": sentence,
            "Annotations": annotations,
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


# Define the function to ground genes in combined results
def ground_genes_in_combined_results(combined_results):
    for entry in combined_results:
        interactions = entry["Combined_Results"]
        if not interactions:  # Check if interactions list is empty
            continue
        genes = set()
        for interaction in interactions:
            parts = interaction.split()
            if len(parts) == 3:
                genes.add(parts[0])
                genes.add(parts[2])
        grounded_genes = ground_genes(list(genes))
        grounded_interactions = []
        for interaction in interactions:
            parts = interaction.split()
            if len(parts) == 3:
                grounded_subject = grounded_genes.get(parts[0], parts[0])
                grounded_object = grounded_genes.get(parts[2], parts[2])
                grounded_interactions.append(f"{grounded_subject} {parts[1]} {grounded_object}")
        entry["Combined_Results"] = grounded_interactions
    return combined_results
