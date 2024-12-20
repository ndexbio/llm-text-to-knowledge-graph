import warnings
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser 
from bel_model import get_bel_extraction_model
warnings.filterwarnings("ignore")


def get_prompt(identifier, filepath):
    # Determine the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the main project directory by going up one level
    project_dir = os.path.dirname(script_dir)

    # Construct the absolute path to the prompt file in the main 'data' directory
    absolute_filepath = os.path.join(project_dir, 'data', filepath)

    with open(absolute_filepath, 'rb') as file:
        content = file.read()

    # Check for BOM and remove it
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]

    lines = content.decode('utf-8').splitlines()
    prompt = []
    capture = False
    for line in lines:
        if line.strip().startswith('#') and identifier in line:
            capture = True
            continue
        if capture:
            if line.strip().startswith('#') and len(prompt) > 0:
                break
            prompt.append(line)
    return ''.join(prompt)


filepath = 'prompt_file_v5.txt'
prompt_identifier = 'general prompt'
prompt = get_prompt(prompt_identifier, filepath)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", prompt),
#     ("human", "{text} | Annotations: {annotations}")
# ])

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("human", "{text}")
])


def initialize_chains(api_key):
    """Initialize extraction chains with the provided API key."""
    bel_extraction_model = get_bel_extraction_model(api_key)

    bel_extraction_chain = prompt | bel_extraction_model | JsonKeyOutputFunctionsParser(key_name="interactions")
    return bel_extraction_chain
