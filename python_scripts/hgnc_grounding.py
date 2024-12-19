import urllib.request
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ground_gene_symbol(text):
    text = text.upper()
    try:
        hgnc_symbol, is_official = get_hgnc_gene_symbol_info(text)

        if hgnc_symbol is not None:
            return (hgnc_symbol, is_official)

        hgnc_symbol, is_official = get_hgnc_gene_symbols_from_alias(text)

        if hgnc_symbol:
            return (hgnc_symbol, is_official)

        # If no match found, return False with empty lists and False flag
        return (False, False)
    except Exception as e:
        logger.error(f"Error in ground_gene_symbol for {text}: {e}")
        return (text, [], [], False)


# This tries to validate an official gene symbol and return information about it, including aliases
def get_hgnc_gene_symbol_info(gene_symbol):
    url = f"https://rest.genenames.org/fetch/symbol/{gene_symbol}"
    headers = {"Accept": "application/json"}

    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            result = json.loads(response.read().decode())
            if 'response' in result and 'docs' in result['response'] and len(result['response']['docs']) > 0:
                gene_info = result['response']['docs'][0]
                hgnc_symbol = gene_info.get('symbol')
                return (hgnc_symbol, True)
            else:
                logger.info(f"No gene info found for symbol: {gene_symbol}")
                return None, False
    except Exception as e:
        logger.error(f"Error fetching gene symbol info for {gene_symbol}: {e}")
        raise


# This tries to find official gene symbols, given an alias.
def get_hgnc_gene_symbols_from_alias(gene_symbol):
    url = f"https://rest.genenames.org/fetch/alias_symbol/{gene_symbol}"
    headers = {"Accept": "application/json"}

    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            result = json.loads(response.read().decode())
            if 'response' in result and 'docs' in result['response'] and len(result['response']['docs']) > 0:
                gene_info = result['response']['docs'][0]
                hgnc_symbol = gene_info.get('symbol')
                return hgnc_symbol, False
            else:
                logger.info(f"No HGNC symbols found for alias: {gene_symbol}")
                return None, False
    except Exception as e:
        logger.error(f"Error fetching HGNC symbols for alias {gene_symbol}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    test_symbols = ["TP53", "p53", "BRCA1", "NonExistentGene"]
    for symbol in test_symbols:
        grounded_symbol, was_official = ground_gene_symbol(symbol)
        print(f"Input: {symbol}")
        print(f"Grounded: {grounded_symbol}")
        print(f"Is official: {was_official}")
        print("---")
