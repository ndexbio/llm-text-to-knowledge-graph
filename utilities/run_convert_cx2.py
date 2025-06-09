#!/usr/bin/env python3

# Deckhard Options: Import Type: for deckard is "edges"; Deckhard Object Type: "edge"
# Deckhard SQL: SELECT * FROM nodes WHERE object_type = 'object_list'

import argparse
import json

from main_scripts.transform_bel_statements import process_llm_results
from main_scripts.convert_to_cx2 import convert_to_cx2


def main():
    parser = argparse.ArgumentParser(description="Process LLM results and merge with Deckhard template.")
    parser.add_argument(
        "--input_file",
        "-i",
        default="llm_results.json",
        help="Path to the LLM results JSON file (default: llm_results.json)"
    )
    parser.add_argument(
        "--template_file",
        "-t",
        default="deckhard_template.json",
        help="Path to the Deckhard template JSON file (default: deckhard_template.json)"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="merged_output.cx2",
        help="Output filename for the merged CX2 file (default: merged_output.cx2)"
    )
    args = parser.parse_args()

    # Read the LLM results
    with open(args.input_file, 'r') as f:
        llm_results = json.load(f)

    # Process and convert to CX2
    extracted_results = process_llm_results(llm_results)
    cx2_network = convert_to_cx2(extracted_results)

    # Write the CX2 output
    with open(args.output_file, 'w') as f:
        json.dump(cx2_network.to_cx2(), f)

    # Merge with the template
    with open(args.output_file, 'r') as f1:
        data1 = json.load(f1)

    with open(args.template_file, 'r') as f2:
        data2 = json.load(f2)

    if not isinstance(data1, list) or not isinstance(data2, list):
        raise ValueError("Both JSON files must contain a list of objects")

    merged_data = data1 + data2

    # Write back the merged result
    with open(args.output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)


if __name__ == "__main__":
    main()
