#!/usr/bin/env python


import sys
import os
import argparse
import logging
import json
import uuid
import shutil

# hack fix to add the directory where this scripts
# resides to the path. This enables the imports below
# to work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# TODO add sys.path to make this work
from bel_main import validate_pmc_id
from convert_to_cx2 import convert_to_cx2
from pub import get_pubtator_paragraphs, download_pubtator_xml
from sentence_level_extraction import llm_bel_processing
from indra_download_extract import setup_output_directory
from transform_bel_statements import process_llm_results


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

def create_tmpdir(theargs):
    """
    Creates temp directory for hidef output with
    a unique name of format cdhidef_<UUID>

    :param theargs: Holds attributes from argparse
    :type theargs: `:py:class:`argparse.Namespace`
    :return: Path to temp directory
    :rtype: str
    """
    tmpdir = os.path.join(theargs.tempdir, 'llmknowledge_' + str(uuid.uuid4()))
    os.makedirs(tmpdir, mode=0o755)
    return tmpdir


def process_paper(pmc_id, api_key, style_path=None):
    """
    Process a single PMC ID to generate BEL statements and CX2 network.

    Args:
        pmc_id (str): The PubMed Central ID of the article to process.
        api_key (str): OpenAI API key for processing.

    Returns:
        bool: True if processing succeeds, False otherwise.
    """
    try:
        validate_pmc_id(pmc_id)
        logger.info(f"Setting up output directory for {pmc_id}")
        output_dir = setup_output_directory(pmc_id)

        file_path = download_pubtator_xml(pmc_id, output_dir)
        if not file_path:
            logger.error("Aborting process due to download failure.")
            return None

        logger.info("Processing xml file to get text paragraphs")
        paragraphs = get_pubtator_paragraphs(file_path)
        # json.dump({'action': 'paragraphs', 'data': paragraphs}, sys.stdout, indent=2)


        logger.info("Processing paragraphs with LLM-BEL model")
        llm_results = llm_bel_processing(paragraphs, api_key)
        # json.dump({'action': 'llmResults', 'data': llm_results}, sys.stdout, indent=2)

        logger.info("Processing LLM results to generate CX2 network")
        extracted_results = process_llm_results(llm_results)
        cx2_network = convert_to_cx2(extracted_results, style_path=style_path)

        # cytoscape web fails if network lacks network attributes so adding name
        cx2_network.set_name('LLM Text to Knowledge Graph for publication: ' + str(pmc_id))
        logger.info(f"Processing completed successfully for {pmc_id}.")
        return cx2_network
    except ValueError as ve:
        logger.error(ve)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def main(pmc_ids, api_key, tempdir='/tmp', style_path=None):
    """
    Main function to process a list of PMC IDs.

    Args:
        pmc_ids (list of str): A list of PubMed Central IDs to process.
        api_key (str): OpenAI API key for processing.
        tempdir (str): Directory to hold temporary files.
    """
    tmpdir = create_tmpdir(argparse.Namespace(tempdir=tempdir))
    logger.info(f"Temporary directory created at {tmpdir}")
    logger.info('Style Path: ' + str(style_path))
    try:
        success_count = 0
        failure_count = 0
        cx2nets = []
        for pmc_id in pmc_ids:
            logger.info(f"Starting processing for PMC ID: {pmc_id}")
            res = process_paper(pmc_id, api_key, style_path=style_path)

            if res is not None:
                success_count += 1
                cx2nets.append(res.to_cx2())
            else:
                failure_count += 1

        # data structure cytoscape web expects
        # for adding a new network
        # just write to standard out
        newres = [{'action': 'addNetworks',
                   'data': cx2nets}]
        json.dump(newres, sys.stdout, indent=2)

        logger.info(f"Processing completed. Success: {success_count}, Failures: {failure_count}")
        if failure_count > 0:
            return 1
        return 0
    finally:
        logger.info(f"Cleaning up temporary directory at {tmpdir}")
        shutil.rmtree(tmpdir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a list of PMC articles and extract interaction data.")
    parser.add_argument('input',
                        help='Put the letter x here. it is temporarily needed for CytoscapeContainerService')
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,  # Defaults to None if no key is provided
        help="OpenAI API key for processing"
    )
    parser.add_argument(
        "--pmc_ids",
        type=str,
        nargs="+",  # Allows multiple arguments to be passed as a list
        help="PubMed Central IDs of the articles to process (space-separated)."
    )
    
    parser.add_argument('--tempdir', default='/tmp',
                        help='Directory needed to hold files temporarily for processing')
    parser.add_argument('--style_path',
                        help='Path to cx_style.json file')
    
    args = parser.parse_args()

    sys.exit(main(args.pmc_ids, args.api_key, style_path=args.style_path))