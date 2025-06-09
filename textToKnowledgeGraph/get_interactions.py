import warnings
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser 
from .bel_model import get_bel_extraction_model
from importlib.resources import files
warnings.filterwarnings("ignore")


def get_prompt(identifier, filepath):
    # Determine the directory of the package
    content = files("textToKnowledgeGraph").joinpath(filepath).read_text(encoding="utf-8")

    # Check for BOM and remove it
    if content.startswith("\ufeff"):
        content = content.lstrip("\ufeff")

    # split into lines (no .decode needed)
    lines = content.splitlines()
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


def load_prompt(prompt_file="prompt_file_v7.txt",
                prompt_identifier="general prompt") -> str:
    """Return just the system‑prompt text (one string)."""
    return get_prompt(prompt_identifier, prompt_file)


def build_bel_extraction_chain(prompt_text: str, api_key: str = None):
    """
    Creates and returns a BEL (Biological Expression Language) extraction processing chain.

    This function constructs a prompt template for a chat-based language model, using the provided
    system prompt text. The template expects user input containing a text and its annotations.
    The resulting chain processes the input through a BEL extraction model and parses the output
    to extract the "interactions" key from the model's JSON response.

    Args:
        prompt_text (str): The system prompt text to guide the BEL extraction process.

    Returns:
        A composed chain that takes annotated text as input, runs BEL extraction, and outputs
        the extracted interactions in structured format.
    """
    ann_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{text} | Annotations: {annotations}")
    ])

    bel_extraction_model = get_bel_extraction_model(api_key)
    if bel_extraction_model is None:
        raise ValueError("BEL extraction model could not be loaded. Please check your API key or model configuration.")
    else:
        print("BEL extraction model loaded successfully.")
    return ann_prompt | bel_extraction_model | JsonKeyOutputFunctionsParser(key_name="interactions")
