import os
import requests
import logging
from lxml import etree

logger = logging.getLogger(__name__)

def download_pubtator_xml(pmc_id, output_dir):
    """
    Downloads the XML file from PubTator API using PMCID and saves it in the specified output directory.
    """
    # Construct the PubTator API URL
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/pmc_export/biocxml?pmcids={pmc_id}"

    # Make the request
    response = requests.get(url)
    if response.status_code == 200:
        logger.info(f"Successfully downloaded XML for PMCID {pmc_id}.")
        file_path = os.path.join(output_dir, f"pmc{pmc_id}.xml")

        # Save the file in the existing output directory
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        logger.error(f"Failed to download XML for PMCID {pmc_id}. Status code: {response.status_code}")
        return None


def get_pubtator_paragraphs(file_path):
    """
    Extracts paragraphs and their annotations from a PubTator XML file,
    ensuring text represents full paragraphs or abstract-like content.

    - Only includes annotations that have an 'NCBI Homologene' or 'identifier' starting with 'MESH:'.
    - For each annotation, extracts 'entity_text', 'type', and the identifier (either MESH or NCBI Homologene ID).
    - Only includes passages that have at least 2 such annotations.
    - Removes duplicate annotations within a passage.

    :param file_path: Path to the PubTator XML file.
    :return: A dictionary with numbered paragraphs as keys and values containing text and annotations.
    """
    tree = etree.parse(file_path)
    root = tree.getroot()

    paragraphs_dict = {}
    passage_elements = root.findall('.//passage')
    paragraph_number = 0

    for passage in passage_elements:
        # Check section type
        section_type = passage.findtext('infon[@key="section_type"]', "").lower()

        # Include sections that are meaningful (abstracts, paragraphs, introduction, etc.)
        if section_type in ['ref', 'title']:
            continue  # Skip non-content sections

        # Extract the passage text
        text_elem = passage.find('text')
        passage_text = text_elem.text.strip() if text_elem is not None else ""
        if len(passage_text) < 20:  # Skip overly short texts
            continue

        # Extract annotations within the passage
        annotations = []
        annotation_elements = passage.findall('.//annotation')
        seen_annotations = set()
        for annotation in annotation_elements:
            # Build a dictionary of infon elements
            infon_dict = {infon.get('key'): infon.text.strip() for infon in annotation.findall('infon') if infon.text}

            # Check annotation criteria
            identifier = ""
            identifier_type = ""
            if 'NCBI Homologene' in infon_dict:
                identifier = infon_dict['NCBI Homologene']
                identifier_type = 'NCBI Homologene'
            elif 'identifier' in infon_dict and infon_dict['identifier'].startswith('MESH:'):
                identifier = infon_dict['identifier']
                identifier_type = 'MESH'

            if identifier:
                # Extract annotation details
                ann_text = annotation.findtext('text', default="").strip()
                ann_type = infon_dict.get('type', '')

                # Create a unique tuple to check for duplicates
                ann_tuple = (ann_text, ann_type, identifier)
                if ann_tuple not in seen_annotations:
                    seen_annotations.add(ann_tuple)
                    annotations.append({
                        'entity_text': ann_text,
                        'type': ann_type,
                        'identifier_type': identifier_type,
                        'identifier': identifier
                    })

        # Include only sections with at least 2 valid annotations
        if len(annotations) >= 2:
            paragraphs_dict[str(paragraph_number)] = {
                'text': passage_text,
                'annotations': annotations
            }
            paragraph_number += 1

    return paragraphs_dict


def extract_annotations_from_pubtator_xml(file_path):
    """
    Extracts annotations from a PubTator XML file.
    :param file_path: Path to the PubTator XML file.
    :return: A list of annotations.
    """
    tree = etree.parse(file_path)
    root = tree.getroot() 
    annotations = []
    for annotation in root.findall('.//annotation'):
        ann = {
            'id': annotation.get('id'),
            'type': annotation.findtext('infon[@key="type"]'),
            'identifier': annotation.findtext('infon[@key="identifier"]'),
            'offset': int(annotation.find('location').get('offset')),
            'length': int(annotation.find('location').get('length')),
            'text': annotation.findtext('text')
        }
        annotations.append(ann)
    return annotations
