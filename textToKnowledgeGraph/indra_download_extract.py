import os
import json
import os.path
import logging

logger = logging.getLogger(__name__)


def setup_output_directory(pmc_id=None, file_path=None, base_path=None):
    if pmc_id and file_path:
        raise ValueError("Please provide only pmc_id or file_path, not both.")
    if not pmc_id and not file_path:
        raise ValueError("Either pmc_id or file_path must be provided.")

    # Get the absolute path to the directory where the script is running
    if not base_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the main project directory (assuming the script is in a subdirectory of the main project directory)
        base_path = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Determine the folder name
    if pmc_id:
        folder_name = pmc_id
    else:
        # Extract the file name (without extension) from file_path
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        folder_name = file_base

    # Set up the output directory within the 'results' directory of the project
    output_dir = os.path.join(base_path, 'results', folder_name)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def save_to_json(data, filename, output_dir):
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    print(f"File saved successfully to {file_path}")
