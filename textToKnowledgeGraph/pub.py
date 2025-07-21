import os
import requests
import logging
from lxml import etree


def download_pubtator_xml(pmc_id, output_dir):
    """
    Downloads the XML file from PubTator API using PMCID and saves it in the specified output directory.
    """
    # Construct the PubTator API URL
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/pmc_export/biocxml?pmcids={pmc_id}"

    # Make the request
    response = requests.get(url)
    if response.status_code == 200:
        logging.info(f"Successfully downloaded XML for PMCID {pmc_id}.")
        file_path = os.path.join(output_dir, f"pmc{pmc_id}.xml")

        # Save the file in the existing output directory
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        logging.error(f"Failed to download XML for PMCID {pmc_id}. Status code: {response.status_code}")
        return None


def get_pubtator_paragraphs(file_path):
    """
    Extracts paragraphs from a PubTator XML file,
    ensuring text represents full paragraphs or abstract-like content.

    Only includes passages that are of sufficient length.

    :param file_path: Path to the PubTator XML file.
    :return: A dictionary with numbered paragraphs as keys and values containing text.
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

        # add the paragraph to the dictionary
        paragraphs_dict[str(paragraph_number)] = {
            'text': passage_text
        }
        paragraph_number += 1

    return paragraphs_dict


def fetch_metadata_via_eutils(article_id):
    """
    Fetches article metadata from NCBI E-Utilities for either a PMID (digits only)
    or a PMCID (e.g. 'PMC123456'), returning four fields:
        1. pmid
        2. title
        3. authors (list of full-name strings)
        4. abstract
        5. doi
        6. journal

    :param article_id: A string like '12345678' (PMID) or 'PMC123456' (PMCID).
    :return: dict with keys {'pmid', 'title', 'authors', 'abstract', 'doi', 'journal'}.
             Some may be None or empty if missing from the record.
    """
    # Detect whether input is PMCID or PMID
    is_pmcid = article_id.upper().startswith("PMC")

    if is_pmcid:
        # For PMCID, remove 'PMC' from the front to get the numeric portion
        numeric_part = article_id.upper().replace("PMC", "")
        params = {
            'db': 'pmc',        # Tells E-Utilities to look in the PMC database
            'id': numeric_part, 
            'rettype': 'full',
            'retmode': 'xml'
        }
    else:
        # Otherwise assume it's a PMID
        params = {
            'db': 'pubmed',     # Tells E-Utilities to look in the PubMed database
            'id': article_id,
            'retmode': 'xml'
        }

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    response = requests.get(url, params=params)

    # Build the return dict with the four fields
    metadata = {
        'pmid': None,
        'title': None,
        'authors': [],
        'abstract': None,
        'doi': None,
        'journal': None
    }

    # If request fails, just return the empty structure
    if response.status_code != 200:
        print(f"E-utilities request failed (HTTP {response.status_code}).")
        return metadata

    # Parse the returned XML
    tree = etree.fromstring(response.content)

    if not is_pmcid:
        pubmed_article = tree.find(".//PubmedArticle")
        if pubmed_article is None:
            # Possibly no results or invalid ID
            return metadata

        # PMID
        pmid_elem = pubmed_article.find(".//PMID")
        if pmid_elem is not None and pmid_elem.text:
            raw = pmid_elem.text.strip()
            metadata['pmid'] = f"pmid{raw}"

        # Title
        title_elem = pubmed_article.find(".//ArticleTitle")
        if title_elem is not None and title_elem.text:
            metadata['title'] = title_elem.text.strip()

        # Authors
        for author_elem in pubmed_article.findall(".//AuthorList/Author"):
            last = author_elem.findtext("LastName")
            fore = author_elem.findtext("ForeName")
            if last or fore:
                full_name = " ".join([fore or "", last or ""]).strip()
                metadata['authors'].append(full_name)

        # Abstract (can have multiple <AbstractText>)
        abstract_texts = pubmed_article.findall(".//Abstract/AbstractText")
        if abstract_texts:
            combined = " ".join(elem.text for elem in abstract_texts if elem.text)
            metadata['abstract'] = combined.strip() if combined else None

        # DOI
        doi_elem = pubmed_article.find(".//ArticleId[@IdType='doi']")
        if doi_elem is not None and doi_elem.text:
            metadata['doi'] = doi_elem.text.strip()

        # Journal
        journal_elem = pubmed_article.find(".//Journal/Title")
        if journal_elem is not None and journal_elem.text:
            metadata['journal'] = journal_elem.text.strip()

    else:
        article_elem = tree.find(".//article")
        if article_elem is None:
            return metadata

        # Extract the PMID from <article-id pub-id-type="pmid">
        for article_id_elem in article_elem.findall(".//article-id"):
            id_type = article_id_elem.get("pub-id-type")
            if id_type == "pmid" and article_id_elem.text:
                raw = article_id_elem.text.strip()
                metadata['pmid'] = f"pmid{raw}" 
            elif id_type == "doi":  
                metadata['doi'] = article_id_elem.text.strip() if article_id_elem.text else None

        # Title
        title_elem = article_elem.find(".//title-group/article-title")
        if title_elem is not None and title_elem.text:
            metadata['title'] = title_elem.text.strip()

        # Authors
        for contrib in article_elem.findall('.//contrib-group/contrib[@contrib-type="author"]'):
            surname = contrib.findtext("name/surname")
            given_names = contrib.findtext("name/given-names")
            if surname or given_names:
                full_name = " ".join([given_names or "", surname or ""]).strip()
                metadata['authors'].append(full_name)

        # Abstract
        abstract_elem = article_elem.find(".//abstract")
        if abstract_elem is not None:
            # Some abstracts contain multiple <p> child elements
            paragraphs = abstract_elem.findall(".//p")
            if paragraphs:
                combined = " ".join(p.text for p in paragraphs if p.text)
                metadata['abstract'] = combined.strip() if combined else None
            else:
                # or a single text node if no <p> children
                if abstract_elem.text:
                    metadata['abstract'] = abstract_elem.text.strip()

        # Journal            
        journal_title_elem = article_elem.find(".//journal-title")
        if journal_title_elem is not None and journal_title_elem.text:
            metadata['journal'] = journal_title_elem.text.strip()

    return metadata
