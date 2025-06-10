#!/usr/bin/env python

import sys
import os
import re
import argparse
import logging
import json
import uuid
import shutil
import time
import traceback
import contextlib
import textToKnowledgeGraph
from textToKnowledgeGraph.convert_to_cx2 import convert_to_cx2
from textToKnowledgeGraph.pub import get_pubtator_paragraphs, download_pubtator_xml, fetch_metadata_via_eutils
from textToKnowledgeGraph.process_text_file import process_paper
from textToKnowledgeGraph.sentence_level_extraction import llm_bel_processing
from textToKnowledgeGraph.grounding_genes import annotate_paragraphs_in_json, process_annotations
from textToKnowledgeGraph.indra_download_extract import save_to_json, setup_output_directory
from textToKnowledgeGraph.transform_bel_statements import process_llm_results

# hack fix to add the directory where this scripts
# resides to the path. This enables the imports above
# to work
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',
                    stream=sys.stderr)
logger = logging.getLogger(__name__)


def validate_pmc_id(pmc_id):
    pattern = r'^PMC\d+$'
    if not re.match(pattern, pmc_id):
        raise ValueError("Invalid PMC ID format. It should start with 'PMC' followed by digits.")


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


def process_pmc_document(
    pmc_id,
    style_path=None,
    api_key=None,
    prompt_file="prompt_file_v6.txt",
    prompt_identifier="general prompt"
):
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

    validate_pmc_id(pmc_id)
    logger.info(f"Setting up output directory for {pmc_id}")
    output_dir = setup_output_directory(pmc_id)

    file_path = download_pubtator_xml(pmc_id, output_dir)
    if not file_path:
        logger.error("Aborting process due to download failure.")
        return None

    logger.info("Processing xml file to get text paragraphs")
    paragraphs = get_pubtator_paragraphs(file_path)
    annotated_paragraphs = annotate_paragraphs_in_json(paragraphs)
    # json.dump({'action': 'paragraphs', 'data': paragraphs}, sys.stdout, indent=2)

    logger.info("Processing paragraphs with LLM-BEL model")
    llm_results = llm_bel_processing(annotated_paragraphs, prompt_file=prompt_file,
                                     prompt_identifier=prompt_identifier, api_key=api_key)
    # json.dump({'action': 'llmResults', 'data': llm_results}, sys.stdout, indent=2)

    logging.info("Processing annotations to extract node URLs")
    processed_annotations = process_annotations(llm_results)

    logger.info("Processing LLM results to generate CX2 network")
    extracted_results = process_llm_results(llm_results)
    cx2_network = convert_to_cx2(extracted_results, style_path=style_path)

    # cytoscape web fails if network lacks network attributes so adding name
    # Fetch metadata for the publication
    metadata = fetch_metadata_via_eutils(pmc_id)
    first_author_last = "Unknown"
    if metadata['authors']:
        # Use the last token of the first author's name (e.g. "Lu" from "Wen‐Cheng Lu")
        first_author_last = metadata['authors'][0].split()[-1]
    pmid_str = metadata['pmid'] or "UnknownPMID"

    # Build description with a blank line between title and abstract
    description_val = f"{metadata['title']}\n\n{metadata['abstract']}"

    # Build reference: if DOI is available, create an HTML snippet with a clickable DOI; otherwise, just show PMID.
    doi_str = metadata.get('doi')
    journal_str = metadata.get('journal', 'Unknown Journal')
    if doi_str:
        reference_val = (
            f"<div>{first_author_last} et al.</div>"
            f"<div><b>{journal_str}</b></div>"
            f"<div><span><a href=\"https://doi.org/{doi_str}\" target=\"_blank\">doi: {doi_str}</a></span></div>"
        )
    else:
        reference_val = f"PMID: {pmid_str}"

    # Set network attributes accordingly:
    cx2_network.set_network_attributes({
        "name": f"{first_author_last} et al.: {pmid_str}",
        "description": description_val,
        "reference": reference_val
    })
    logger.info(f"Processing completed successfully for {pmc_id}.")
    return cx2_network


def main(pmc_ids, api_key, tempdir='/tmp', style_path=None,
         prompt=None):
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
        with contextlib.redirect_stdout(sys.stderr):
            for pmc_id in pmc_ids:
                logger.info(f"Starting processing for PMC ID: {pmc_id}")
                res = process_pmc_document(pmc_id, style_path=style_path,
                                           api_key=api_key,
                                           prompt_file=prompt)
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
        required=True,
        help="Space-separated list of PMC IDs (e.g. PMC123456 PMC234567)."
    )

    parser.add_argument('--tempdir', default='/tmp',
                        help='Directory needed to hold files temporarily for processing')
    parser.add_argument('--style_path', default=os.path.join(textToKnowledgeGraph.__path__[0],
                                                             'cx_style.json'),
                        help='Path to CX2 Network file with style to use')
    parser.add_argument('--prompt',
                        default=os.path.join(textToKnowledgeGraph.__path__[0],
                                             'prompt_file_v7.txt'),
                        help='Path to alternate prompt file. Default is '
                             'to use internal prompt')
    args = parser.parse_args()

    try:
        sys.exit(main(re.split(r'\s+', args.pmc_ids), args.api_key, style_path=args.style_path,
                      prompt=args.prompt))
    except Exception as e:
        sys.stderr.write('\n\nCaught exception: ' + str(e))
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(2)
