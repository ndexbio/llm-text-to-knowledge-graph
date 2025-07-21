import pandas as pd
import re
import os
from ndex2.cx2 import PandasDataFrameToCX2NetworkFactory
from ndex2.cx2 import RawCX2NetworkFactory
import logging

logger = logging.getLogger(__name__)


def extract_label(bel_expression: str) -> str:
    """
    Extract a human-friendly label from a BEL expression.
    Handles nested functions: multiple terms, modifications, locations.
    Joins all found names with commas.
    """
    # Find all namespace:value terms, with or without quotes
    ns_terms = re.findall(r':[\"\']?([^\"\')]+)[\"\']?', bel_expression)

    # Find protein modifications, if any
    mods = re.findall(r'pmod\(([a-zA-Z]+)', bel_expression)

    # Find location names, if any
    locs = re.findall(r'loc\([a-zA-Z]+:([^\)]+)\)', bel_expression)
    locs = [loc.replace('"', '').replace("'", "") for loc in locs]

    # Collect parts
    parts = []
    if ns_terms:
        parts.extend([t.strip() for t in ns_terms])
    if mods:
        parts.extend(mods)
    if locs:
        parts.extend(locs)

    # Join all parts with commas
    if parts:
        return ', '.join(parts)
    return bel_expression


def extract_type(bel_expression):
    """
    Extracts the type from a BEL expression, i.e., the prefix before the first '('.
    """
    match = re.match(r'([a-zA-Z]+)\(', bel_expression)
    return match.group(1) if match else "unknown"  # Default to "unknown" if no match


def add_style_to_network(cx2_network=None, style_path=None):
    """
    Adds ctyle from CX2Network set in **style_path** to
    **cx2network**
    """
    if style_path is not None and os.path.isfile(style_path):
        logger.info('Setting visual style properties')
        cx2netfac = RawCX2NetworkFactory()
        style_cx_net = cx2netfac.get_cx2network(style_path)
        cx2_network.set_visual_properties(style_cx_net.get_visual_properties())


def convert_to_cx2(extracted_results, style_path=None):
    # Initialize lists to store extracted data
    source_list = []
    target_list = []
    interaction_list = []
    source_label_list = []
    target_label_list = []
    bel_expression_list = []
    text_list = []
    evidence_list = []

    # Create mappings for node names to unique integer IDs
    node_name_to_id = {}
    node_id_counter = 0  # Start counter for unique node IDs

    annotation_map = {}
    for entry in extracted_results:
        if "entry_name" in entry and "url" in entry:
            name = entry["entry_name"]
            if name not in annotation_map:
                annotation_map[name] = entry["url"]

    # Extract data and build node mappings
    for entry in extracted_results:
        source = entry.get("source")
        interaction = entry.get("interaction")
        target = entry.get("target")
        text = entry.get("text")
        evidence = entry.get("evidence")

        if source and interaction is None and target is None:
            if source not in node_name_to_id:
                node_name_to_id[source] = node_id_counter
                node_id_counter += 1
            continue

        if target and interaction is None and source is None:
            if target not in node_name_to_id:
                node_name_to_id[target] = node_id_counter
                node_id_counter += 1
            continue

        if source and interaction and target:
            # Assign unique integer ID if node doesn't exist in mapping
            if source not in node_name_to_id:
                node_name_to_id[source] = node_id_counter
                node_id_counter += 1
            if target not in node_name_to_id:
                node_name_to_id[target] = node_id_counter
                node_id_counter += 1

            # Extract labels for source and target
            source_label_list.append(extract_label(source))
            target_label_list.append(extract_label(target))
            bel_expression_list.append(f"{source} {interaction} {target}")
            text_list.append(text)
            source_list.append(source)
            target_list.append(target)
            interaction_list.append(interaction)
            evidence_list.append(evidence)

    # Create a DataFrame for CX2 conversion
    df = pd.DataFrame({
        'source': source_list,
        'target': target_list,
        'interaction': interaction_list,
        'source_label': source_label_list,
        'target_label': target_label_list,
        'bel_expression': bel_expression_list,
        'text': text_list,
        'evidence': evidence_list
    })

    # Convert DataFrame to CX2 network format
    factory = PandasDataFrameToCX2NetworkFactory()
    cx2_network = factory.get_cx2network(df, source_field='source', 
                                         target_field='target', edge_interaction='interaction')

    # Add visual properties style to network
    add_style_to_network(cx2_network=cx2_network,
                         style_path=style_path)

    # Retrieve nodes from the CX2 structure
    cx2_structure = cx2_network.to_cx2()

    nodes_aspect = next((aspect for aspect in cx2_structure if isinstance(aspect, dict) and "nodes" in aspect), None)
    existing_node_ids = {node['id'] for node in nodes_aspect["nodes"]} if nodes_aspect else set()

    # Add each node with its integer ID and set attributes if not already added
    for node_name, node_id in node_name_to_id.items():
        # Only add node if it does not already exist in the network
        if node_id not in existing_node_ids:
            cx2_network.add_node(node_id)

        # Extract label and type from the node name
        label = extract_label(node_name)
        node_type = extract_type(node_name)
        node_url = annotation_map.get(node_name, annotation_map.get(label))

        # Set node attributes: only name, label, and type.
        cx2_network.set_node_attribute(node_id, 'name', node_name)
        cx2_network.set_node_attribute(node_id, 'label', label)
        cx2_network.set_node_attribute(node_id, 'type', node_type)
        if node_url:
            cx2_network.set_node_attribute(node_id, 'id', node_url)

    cx2_network._cx2 = cx2_structure

    return cx2_network
