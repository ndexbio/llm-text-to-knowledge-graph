# import requests
import os
import json
from indra.sources import reach


def process_nxml_file(file_name, output_dir, url='http://localhost:8080/api/uploadFile'):
    """
    Process an NXNML file using a local REACH API server.
    :param file_name: Name of the file to process.
    :param url: URL of the local REACH service.
    :return: Processed results or None if failed.
    """
    try:

        # Process the file using Reach
        rp = reach.process_nxml_file(file_name=file_name, url=url)

        if rp and hasattr(rp, 'statements'):
            # Convert statements to JSON
            statements_json = [stmt.to_json() for stmt in rp.statements]     
            # Extract the base file name (without the '.xml') for the output JSON file name
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            output_fname = f"{output_dir}/{base_name}.json"    
            # Save the results as a JSON file in the specified output directory
            with open(output_fname, 'w') as f:
                json.dump(statements_json, f, indent=4)        
            print(f'Results saved to {output_fname}')
            return statements_json
        else:
            print('No results returned from REACH processing.')
            return None

    except Exception as e:
        print(f'Failed to process file due to an error: {e}')
        return None
