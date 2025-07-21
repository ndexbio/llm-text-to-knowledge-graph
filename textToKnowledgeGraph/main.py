import sys
import os
import re
import time
import argparse
import logging
from ndex2.client import Ndex2
from .convert_to_cx2 import convert_to_cx2
from .pub import get_pubtator_paragraphs, download_pubtator_xml, fetch_metadata_via_eutils
from .process_text_file import process_paper
from .grounding_genes import annotate_paragraphs_in_json, process_annotations
from .sentence_level_extraction import llm_bel_processing
from .indra_download_extract import save_to_json, setup_output_directory
from .transform_bel_statements import process_llm_results


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def validate_pmc_id(pmc_id):
    pattern = r'^PMC\d+$'
    if not re.match(pattern, pmc_id):
        raise ValueError("Invalid PMC ID format. It should start with 'PMC' followed by digits.")


def process_pmc_document(pmc_id, 
                         api_key, 
                         ndex_email=None, 
                         ndex_password=None, 
                         style_path=None, 
                         upload_to_ndex=False,
                         prompt_file="prompt_file_v7.txt",
                         prompt_identifier="general prompt"):
    """
    Process a document given a PMC ID.
    Steps:
      1. Validate PMC ID and set up the output directory.
      2. Download the PubTator XML and extract paragraphs.
      3. Annotate paragraphs using Gilda (passing JSON data directly).
      4. Process annotated paragraphs through the LLM-BEL model.
      5. Process annotations to generate node URLs.
      6. Convert results to a CX2 network and optionally upload to NDEx.
    """
    try:
        validate_pmc_id(pmc_id)
        logging.info(f"Setting up output directory for {pmc_id}")
        output_dir = setup_output_directory(pmc_id)

        file_path = download_pubtator_xml(pmc_id, output_dir)
        if not file_path:
            logging.error("Aborting process due to download failure.")
            return False

        logging.info("Processing XML file to extract text paragraphs")
        paragraphs = get_pubtator_paragraphs(file_path)
        paragraphs_filename = f"{pmc_id}_pub_paragraphs.json"
        save_to_json(paragraphs, paragraphs_filename, output_dir)

        logging.info("Annotating paragraphs using Gilda")
        annotated_paragraphs = annotate_paragraphs_in_json(paragraphs)
        annotated_filename = f"{pmc_id}_annotated_paragraphs.json"
        save_to_json(annotated_paragraphs, annotated_filename, output_dir)

        logging.info("Processing annotated paragraphs with the LLM-BEL model")
        llm_results = llm_bel_processing(annotated_paragraphs, api_key, 
                                         prompt_file=prompt_file, prompt_identifier=prompt_identifier)
        llm_filename = 'llm_results.json'
        save_to_json(llm_results, llm_filename, output_dir)

        logging.info("Processing annotations to extract node URLs")
        processed_annotations = process_annotations(llm_results)

        logging.info("Generating CX2 network")
        extracted_results = process_llm_results(processed_annotations)
        cx2_network = convert_to_cx2(extracted_results, style_path=style_path)

        # Fetch metadata for the publication
        metadata = fetch_metadata_via_eutils(pmc_id)
        first_author_last = "Unknown"
        if metadata['authors']:  
            # e.g. if authors[0] == "Wen‐Cheng Lu", last name is "Lu"
            first_author_last = metadata['authors'][0].split()[-1]
        pmid_str = metadata['pmid'] or "UnknownPMID"

        cx2_network.set_network_attributes({
            "name": f"{first_author_last} et al.: {pmid_str}",
            "description": f"{metadata['title']}, {metadata['abstract']}",
            "reference": f"PMID: {pmid_str}"
        })
        cx2_filename = 'cx2_network.cx'
        save_to_json(cx2_network.to_cx2(), cx2_filename, output_dir)

        if upload_to_ndex:
            if not ndex_email or not ndex_password:
                logging.error("NDEx email and password are required to upload.")
                return False
            logging.info("Uploading CX2 network to NDEx")
            client = Ndex2(username=ndex_email, password=ndex_password)
            client.save_new_cx2_network(cx2_network.to_cx2())

        logging.info(f"Processing completed successfully for {pmc_id}.")
        return True

    except ValueError as ve:
        logging.error(ve)
        sys.exit(1)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
        return False


def process_file_document(file_path, api_key, pmid_or_pmcid=None, custom_name=None, 
                          ndex_email=None, 
                          ndex_password=None, 
                          prompt_file="prompt_file_v7.txt",
                          prompt_identifier="general prompt",
                          style_path=None, upload_to_ndex=False):
    """
    Process a document given a file path (PDF or TXT).
    Steps:
      1. Set up the output directory using the file's base name.
      2. Process the file to extract paragraphs.
      3. Annotate paragraphs using Gilda (passing JSON data directly).
      4. Process annotated paragraphs with the LLM-BEL model.
      5. Process annotations to generate node URLs.
      6. Convert results to a CX2 network and optionally upload to NDEx.
    """
    try:
        output_name = os.path.splitext(os.path.basename(file_path))[0]
        logging.info(f"Setting up output directory for file: {output_name}")
        output_dir = setup_output_directory(output_name)

        logging.info("Processing file to extract paragraphs")
        paragraphs = process_paper(file_path)
        paragraphs_filename = "processed_paragraphs.json"
        save_to_json(paragraphs, paragraphs_filename, output_dir)

        logging.info("Annotating paragraphs using Gilda")
        annotated_paragraphs = annotate_paragraphs_in_json(paragraphs)
        annotated_filename = "annotated_paragraphs.json"
        save_to_json(annotated_paragraphs, annotated_filename, output_dir)

        logging.info("Processing annotated paragraphs with the LLM-BEL model")
        llm_results = llm_bel_processing(annotated_paragraphs, api_key, 
                                         prompt_file=prompt_file, prompt_identifier=prompt_identifier)
        llm_filename = 'llm_results.json'
        save_to_json(llm_results, llm_filename, output_dir)

        logging.info("Processing annotations to extract node URLs")
        processed_annotations = process_annotations(llm_results)

        logging.info("Generating CX2 network")
        extracted_results = process_llm_results(processed_annotations)
        cx2_network = convert_to_cx2(extracted_results, style_path=style_path)

        if pmid_or_pmcid:
            # use E-Utilities to get metadata -> "FirstAuthor et al.: PMID"
            metadata = fetch_metadata_via_eutils(pmid_or_pmcid)
            first_author_last = "Unknown"
            if metadata['authors']:  
                # e.g. if authors[0] == "Wen‐Cheng Lu", last name is "Lu"
                first_author_last = metadata['authors'][0].split()[-1]
            pmid_str = metadata['pmid'] or "UnknownPMID"

            cx2_network.set_network_attributes({
                "name": f"{first_author_last} et al.: {pmid_str}",
                "description": f"{metadata['title']}, {metadata['abstract']}",
                "reference": f"PMID: {pmid_str}"
            })
        elif custom_name:
            cx2_network.set_name(custom_name)
        else:
            timestamp_str = time.strftime("%Y%m%d_%H%M")
            cx2_network.set_name(f"LLM Text to Knowledge Graph: {timestamp_str}")

        cx2_filename = 'cx2_network.cx'
        save_to_json(cx2_network.to_cx2(), cx2_filename, output_dir)

        if upload_to_ndex:
            if not ndex_email or not ndex_password:
                logging.error("NDEx email and password are required to upload.")
                return False
            logging.info("Uploading CX2 network to NDEx")
            client = Ndex2(username=ndex_email, password=ndex_password)
            client.save_new_cx2_network(cx2_network.to_cx2())

        logging.info(f"Processing completed successfully for {output_name}.")
        return True

    except ValueError as ve:
        logging.error(ve)
        sys.exit(1)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
        return False


def main(
    api_key,
    pmc_ids=None,
    pdf_paths=None,
    txt_paths=None,
    ndex_email=None,
    ndex_password=None,
    style_path=None,
    upload_to_ndex=False,
    prompt_file="prompt_file_v7.txt",
    prompt_identifier="general prompt",
    custom_name=None,
    pmid_for_file=None
):
    """
    Determines the input type (PMC ID or file path) and routes to the appropriate processing function.
    Exactly one of pmc_id, pdf_path, or txt_path should be provided.
    """
    # Default empty lists if None
    pmc_ids = pmc_ids or []
    pdf_paths = pdf_paths or []
    txt_paths = txt_paths or []

    # Check if at least one input is provided
    total_inputs = len(pmc_ids) + len(pdf_paths) + len(txt_paths)
    if total_inputs == 0:
        raise ValueError("No inputs provided. Provide at least one PMC ID, PDF path, or TXT path.")

    overall_success = True

    # Process each PMC ID
    for pmc_id in pmc_ids:
        success = process_pmc_document(
            api_key=api_key,
            pmc_id=pmc_id,
            ndex_email=ndex_email,
            ndex_password=ndex_password,
            style_path=style_path,
            upload_to_ndex=upload_to_ndex,
            prompt_file=prompt_file,
            prompt_identifier=prompt_identifier
        )
        if not success:
            logging.error(f"Processing failed for PMC ID: {pmc_id}")
            overall_success = False
        else:
            logging.info(f"Processing completed successfully for PMC ID: {pmc_id}")

    # Process each PDF file
    for pdf_path in pdf_paths:
        success = process_file_document(
            api_key=api_key,
            file_path=pdf_path,
            ndex_email=ndex_email,
            ndex_password=ndex_password,
            style_path=style_path,
            upload_to_ndex=upload_to_ndex,
            custom_name=custom_name,
            pmid_or_pmcid=pmid_for_file,
            prompt_file=prompt_file,
            prompt_identifier=prompt_identifier
        )
        if not success:
            logging.error(f"Processing failed for file: {pdf_path}")
            overall_success = False
        else:
            logging.info(f"Processing completed successfully for file: {pdf_path}")

    # Process each TXT file
    for txt_path in txt_paths:
        success = process_file_document(
            api_key=api_key,
            file_path=txt_path,
            ndex_email=ndex_email,
            ndex_password=ndex_password,
            style_path=style_path,
            upload_to_ndex=upload_to_ndex,
            custom_name=custom_name,
            pmid_or_pmcid=pmid_for_file,
            prompt_file=prompt_file,
            prompt_identifier=prompt_identifier
        )
        if not success:
            logging.error(f"Processing failed for file: {txt_path}")
            overall_success = False
        else:
            logging.info(f"Processing completed successfully for file: {txt_path}")

    return overall_success


def cli() -> None:
    """Command-line interface for textToKnowledgeGraph."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # No need to go up one level if cx_style.json is right here
    default_style_path = os.path.join(script_dir, "cx_style.json")

    # Configure the argument parser
    parser = argparse.ArgumentParser(
        description="Process multiple documents to extract BEL statements and generate a CX2 network."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key for processing"
    )
    parser.add_argument(
        "--pmc_ids",
        nargs="*",
        default=[],
        help="Space-separated list of PMC IDs (e.g. PMC123456 PMC234567)."
    )
    parser.add_argument(
        "--pdf_paths",
        nargs="*",
        default=[],
        help="Space-separated list of PDF file paths."
    )
    parser.add_argument(
        "--txt_paths",
        nargs="*",
        default=[],
        help="Space-separated list of text file paths."
    )
    parser.add_argument(
        "--ndex_email",
        type=str,
        required=False,
        help="NDEx account email for authentication."
    )
    parser.add_argument(
        "--ndex_password",
        type=str,
        required=False,
        help="NDEx account password for authentication."
    )
    parser.add_argument(
        "--upload_to_ndex",
        action="store_true",
        help="Set this flag to upload the CX2 network to NDEx."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt_file_v7.txt",
        help="Path to a custom LLM prompt file (defaults to prompt_file_v7.txt)")
    parser.add_argument(
        "--prompt_identifier",
        type=str,
        default="general prompt",
        help="Block identifier inside the prompt file.")
    parser.add_argument(
        "--style_path",
        type=str,
        default=default_style_path,
        help="Path to the JSON file containing the Cytoscape visual style."
    )
    parser.add_argument(
        "--custom_name", 
        type=str, 
        help="If provided (and you're processing a file, not a PMC ID), "
        "use this name for the network instead of timestamp."
    )
    parser.add_argument(
        "--pmid_for_file", 
        type=str,
        help="If you're processing a file, and it corresponds to a known article, "
        "supply a PMID or PMCID so we can fetch metadata for naming."
    )

    args = parser.parse_args()

    # Configure logging & measure start time
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    start_time = time.time()

    # Check that at least one input was provided
    total_inputs = len(args.pmc_ids) + len(args.pdf_paths) + len(args.txt_paths)
    if total_inputs == 0:
        logging.error("No inputs provided. Provide at least one PMC ID, PDF path, or TXT path.")
        sys.exit(1)

    overall_success = True

    # Process each PMC ID
    for pmc_id in args.pmc_ids:
        logging.info(f"Starting processing for PMC ID: {pmc_id}")
        success = process_pmc_document(
            api_key=args.api_key,
            pmc_id=pmc_id,
            ndex_email=args.ndex_email,
            ndex_password=args.ndex_password,
            style_path=args.style_path,
            upload_to_ndex=args.upload_to_ndex,
            prompt_file=args.prompt_file,
            prompt_identifier=args.prompt_identifier
        )
        if not success:
            logging.error(f"Processing failed for PMC ID: {pmc_id}")
            overall_success = False

    # Process each PDF file
    for pdf_path in args.pdf_paths:
        logging.info(f"Starting processing for PDF: {pdf_path}")
        success = process_file_document(
            api_key=args.api_key,
            file_path=pdf_path,
            ndex_email=args.ndex_email,
            ndex_password=args.ndex_password,
            style_path=args.style_path,
            upload_to_ndex=args.upload_to_ndex,
            custom_name=args.custom_name,
            pmid_or_pmcid=args.pmid_for_file,
            prompt_file=args.prompt_file,
            prompt_identifier=args.prompt_identifier
        )
        if not success:
            logging.error(f"Processing failed for PDF: {pdf_path}")
            overall_success = False

    # Process each TXT file
    for txt_path in args.txt_paths:
        logging.info(f"Starting processing for TXT: {txt_path}")
        success = process_file_document(
            api_key=args.api_key,
            file_path=txt_path,
            ndex_email=args.ndex_email,
            ndex_password=args.ndex_password,
            style_path=args.style_path,
            upload_to_ndex=args.upload_to_ndex,
            custom_name=args.custom_name,
            pmid_or_pmcid=args.pmid_for_file,
            prompt_file=args.prompt_file,
            prompt_identifier=args.prompt_identifier
        )
        if not success:
            logging.error(f"Processing failed for TXT: {txt_path}")
            overall_success = False

    elapsed_time = time.time() - start_time
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

    if not overall_success:
        logging.error("One or more documents failed to process.")
        sys.exit(1)
    else:
        logging.info("Batch processing completed successfully.")


if __name__ == "__main__":
    cli()
