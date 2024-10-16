import json
from get_interactions import extraction_chain, bel_extraction_chain
from indra.sources import reach
import time
from grounding_genes import ground_genes


def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def extract_sentences(data):
    sentences = {}
    index = 0  # Initialize a counter to keep track of the indices
    for item in data:
        if 'evidence' in item:
            for evidence in item['evidence']:
                if 'text' in evidence:
                    sentences[str(index)] = evidence['text']
                    index += 1  # Increment the index for each sentence
    return sentences


# Initialize dictionaries to store the results
llm_results = {}
indra_reach_results = {}


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
def llm_bel_processing(sentences):
    llm_results = {"LLM_extractions": []}
    start_time = time.time()

    # Loop through the sentences dictionary directly
    for index, sentence_info in sentences.items():
        sentence = sentence_info['text']  # Assuming 'text' is the correct key for the sentence
        results = bel_extraction_chain.invoke({
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
            "Sentence": sentence,
            "Annotations": annotations,
            "Results": results
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
    return llm_results


# perform extraction using indra reach
def indra_processing(sentences):
    indra_reach_results = {"INDRA_REACH_extractions": []}
    start_time = time.time()

    # Loop through the sentences dictionary directly
    for index, sentence in sentences.items():
        reach_processor = reach.process_text(sentence, url=reach.local_text_url)
        stmts = reach_processor.statements
        statements_json = [stmt.to_json() for stmt in stmts]
        indra_reach_results["INDRA_REACH_extractions"].append({
            "Index": index,
            "Sentence": sentence,
            "Results": statements_json
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Time taken for indra processing: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
    return indra_reach_results


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


#function to save both indra outputs and llm outputs in one file
def combine_llm_and_indra_results(llm_filepath, indra_filepath):
    llm_results = load_json_data(llm_filepath)
    indra_results = load_json_data(indra_filepath)
    combined_data = []

    # Create a dictionary to match entries by index
    llm_dict = {entry["Index"]: entry for entry in llm_results}
    indra_dict = {entry["Index"]: entry for entry in indra_results}

    # Get all unique indices from both LLM and INDRA results
    all_indices = set(llm_dict.keys()).union(set(indra_dict.keys()))

    for index in all_indices:
        llm_entry = llm_dict.get(index, {"Sentence": "", "Combined_Results": []})
        indra_entry = indra_dict.get(index, {"Sentence": "", "Combined_Results": []})

        combined_entry = {
            "index": index,
            "sentence": llm_entry["Sentence"] or indra_entry["Sentence"],
            "llm": llm_entry["Combined_Results"],
            "indra": indra_entry["Combined_Results"]
        }
        combined_data.append(combined_entry)
    return combined_data


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
