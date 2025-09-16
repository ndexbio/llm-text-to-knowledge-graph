import os
import re
from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import OpenAI


def clean_paragraph(text: str) -> str:
    """
    Cleans text by removing unnecessary newlines while preserving sentence structure.
    """
    text = re.sub(r"\n+", " ", text)  # Replace multiple newlines with a space
    return text.strip()  # Remove leading/trailing spaces


def process_text_with_llm(text: str, api_key: str, model="gpt-4o") -> dict:
    """
    Uses GPT-4o to remove unwanted sections and split text into structured paragraphs.
    If api_key is "none", uses environment variables from .env file.
    """
    # Create the prompt first so it's available for both code paths
    prompt = (
        "You are an AI assistant that processes scientific papers.  "
        "Your task is to remove unwanted sections such as 'References', 'Appendix', 'Acknowledgments', "
        " 'Author's list', 'Keywords', while ensuring that all other scientific content is preserved. "
        "Do NOT remove the 'Abstract' section, but do NOT include the word 'Abstract' in the output. "
        "For all section headings (e.g., 'Introduction', 'Methods', 'Results'), remove the heading itself and only "
        " return the content below it.\n\n "
        "For example, if the input is:\n"
        "'Introduction\\n\\nThis study examines...'\n"
        "Your output should be:\n"
        "'This study examines...'\n\n"
        "Each paragraph should be clearly separated by a **double newline (`\\n\\n`)**.\n\n"
        "Do NOT return JSON output. Instead, return paragraphs as plain text with double newlines (`\\n\\n`) "
        "between them.\n\n"
        "Example Output:\n"
        "This study explores...\n\n"
        "The research investigates...\n\n"
        "We used...\n\n"
        "Now, process the following text:\n\n"
        "Input Text:\n" + text
    )

    # Choose between litellm (for "none" API key) and direct OpenAI API
    if api_key == "none":
        # Load from .env in current directory
        #print("Using API key from .env file in current directory")
        #load_dotenv(override=False)

        # Use litellm for completion
        client = OpenAI(api_key="none", base_url="http://0.0.0.0:4000")
        response = client.chat.completions.create(
            model="none",
            messages=[
                {"role": "system", "content": "You are an AI that processes text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10000
        )
    else:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI that processes text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10000
        )

    # Extract the text content from the response
    output_text = response.choices[0].message.content.strip()

    # Process the extracted text
    paragraphs = output_text.split("\n\n")  # Split at double newlines
    structured_data = {str(i + 1): {"text": clean_paragraph(para)} for i, para in enumerate(paragraphs) if para.strip()}

    return structured_data


def process_paper(api_key: str, pdf_path: str = None, txt_path: str = None) -> str:
    """
    Extracts text from a scientific paper (PDF or TXT) using MarkItDown.
    Passes the extracted text to GPT-4o for unwanted section removal and paragraph structuring.
    Saves the final processed text as a JSON file.
    """
    if not pdf_path and not txt_path:
        raise ValueError("Either `pdf_path` or `txt_path` must be provided.")

    if pdf_path and txt_path:
        raise ValueError("Provide only one input: either `pdf_path` or `txt_path`, not both.")

    # Validate the file
    file_path = pdf_path or txt_path
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' does not exist.")

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension not in [".pdf", ".txt", ".pptx", ".xlsx", ".html"]:
        raise ValueError("Unsupported file format. Only .txt and .pdf are allowed.")

    #Extract text using MarkItDown
    md = MarkItDown()
    result = md.convert(file_path)
    extracted_text = result.text_content

    # Get API key - either from environment or use "none" to trigger .env loading
    # Use direct OpenAI client
    if not api_key:
        raise ValueError("Missing API Key. Please provide a valid API key or use 'none' with an .env.")

    # Process the text using the LLM
    structured_paragraphs = process_text_with_llm(extracted_text, api_key)
    return structured_paragraphs
