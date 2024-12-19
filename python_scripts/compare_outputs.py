import json
from model import model

# Load INDRA and LLM extractions
with open('results/pmid9813040/indra_combined_results.json', 'r') as f:
    indra_results = json.load(f)

with open('results/pmid9813040/llm_results.json', 'r') as f:
    llm_results = json.load(f)

# Define LLM comparison prompt
comparison_prompt = """
You will receive a sentence along with two different biological extractions: one from INDRA and one from an LLM model. 
Your task is to:
1. Compare the accuracy of each extraction (INDRA vs. LLM) against the given sentence.
2. Judge which extraction is more accurate, or if they both are correct.
3. Reflect on the biological reasoning, explaining how the biological entities and interactions in the 
more accurate extraction align with biological knowledge.

For each entry, provide the following format:

- Discussion: Discuss the strengths and weaknesses of each extraction.
- Judgment: State which extraction is more accurate or if both are equally accurate.
- Biological Reasoning: Reflect on the biological entities and their interactions in the more accurate 
extraction, explaining how they align with known biological pathways or mechanisms.


Ensure that each section starts on a new line for readability.
"""

# Initialize LLM model
llm_model = model


# Function to compare extractions
def compare_extractions(sentence, indra_extraction, llm_extraction):
    prompt = f"{comparison_prompt}\n\nSentence: {sentence}\nINDRA Extraction: \
    {indra_extraction}\nLLM Extraction: {llm_extraction}\n\nResponse:"
    response = llm_model.invoke(prompt)
    return response.content  # Only extract the text content


# Run comparison and collect results
comparison_results = {}

for index in range(len(indra_results)):
    sentence = indra_results[index]['Sentence']  # Use "Sentence" key
    indra_extraction = indra_results[index].get('Combined_Results', [])
    llm_extraction = llm_results["LLM_extractions"][index]["Results"]
    # Run comparison
    analysis = compare_extractions(sentence, indra_extraction, llm_extraction)
    # Store the result with index
    comparison_results[index + 1] = {
        "Sentence": sentence,
        "INDRA Extraction": indra_extraction,
        "LLM Extraction": llm_extraction,
        "Analysis": analysis.replace("\\n", "\n")
    }

# Save comparison results
with open("pmid9813040_comparison_results.json", "w") as f:
    json.dump(comparison_results, f, indent=4)
print("Comparison analysis saved to 'comparison_results.json'.")
