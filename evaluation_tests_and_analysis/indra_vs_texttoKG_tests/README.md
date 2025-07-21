# INDRA vs TextToKG BEL Extraction Comparison

This folder contains the evaluation pipeline for comparing BEL (Biological Expression Language) statements extracted by our TextToKG LLM-based tool against those produced by the INDRA REACH pipeline. The comparison quantifies overlap, assesses component-level matches, and provides both automated scoring and LLM-based semantic similarity ratings.

## Overview

The evaluation follows a three-step process:

1. **Convert** INDRA REACH output to BEL format
2. **Compare** INDRA and TextToKG BEL statements at multiple levels
3. **Analyze** results with statistical summaries and visualizations

## Files

### Core Scripts

#### `indra_to_bel.py`

Converts INDRA REACH JSON output into BEL format for comparison.

**Key Features:**

- Parses INDRA statements using `stmts_from_json`
- Builds BEL graphs using PyBEL assembler
- Outputs both node-link graph format and flattened statement format
- Maps BEL statements to their evidence texts

**Usage:**

```bash
python indra_to_bel.py <indra_reach_results.json>
```

**Outputs:**

- `indra_bel.nodelink.json` - Full BEL graph in node-link format
- `indra_bel_statements.json` - Flattened BEL statements with evidence mapping

#### `indra_vs_llm.py`

Main comparison script that analyzes matches between INDRA and TextToKG outputs.

**Key Features:**

- **BEL Statement Parser**: Decomposes statements into subject, relationship, and object components
- **Namespace Normalization**: Converts between INDRA format (`HGNC:391 ! AKT1`) and LLM format (`HGNC:AKT1`)
- **Modification Normalization**: Maps verbose terms (e.g., "protein phosphorylation") to BEL abbreviations (e.g., "Ph")
- **Multi-level Matching**:
  - Component-level: Subject, relationship, and object matching
  - Namespace-level: Overlap in biological identifiers
  - Semantic-level: LLM-based similarity assessment
- **Scoring Algorithm**: Weighted scoring based on component matches (relationship: 40%, subject: 25%, object: 25%, string similarity: 10%)

**Usage:**

```bash
export OPENAI_API_KEY="your_api_key_here"
python indra_vs_llm.py
```

**Note:** The script currently has hardcoded paths. Modify lines in `main()` function to match your file locations.

### Analysis Notebook

#### `bel-analysis-notebook.ipynb`

Jupyter notebook for visualizing and analyzing comparison results.

**Analyses Included:**

- Distribution of match types (Exact, Best Match, LLM Only, INDRA Only)
- Component-level match statistics
- Similarity rating distributions
- Score histograms and correlations

### Data Files

#### Input Files (Required)

- `indra_bel_cleaned.json` - Cleaned INDRA BEL statements
- `texttoKG_cleaned.json` - Cleaned TextToKG LLM output

#### Output Files (Generated)

- `bel_comparison_results.csv` - Detailed comparison results (74 rows, 18 columns) with the following columns:
  - `index`, `evidence` - Document/paragraph identifiers
  - `llm_statement`, `indra_statement` - Original BEL statements
  - `llm_subject/relationship/object`, `indra_subject/relationship/object` - Parsed components
  - `subject_match`, `relationship_match`, `object_match` - Boolean match flags
  - `subject_namespace_match`, `object_namespace_match` - Namespace overlap flags
  - `match_score` - Numerical similarity score (0-1)
  - `similarity_rating` - LLM-assessed similarity (Good/Medium/Bad/INDRA Only)
  - `match_type` - Classification (Best Match/LLM Only/INDRA Only)

## Installation

```bash
# Core dependencies
pip install indra[bel] pybel

# Comparison script dependencies
pip install pandas python-dotenv openai

# Analysis notebook dependencies
pip install jupyter matplotlib seaborn
```

## Workflow

1. **Prepare Input Data**
   - Ensure you have INDRA REACH results in JSON format
   - Ensure TextToKG results are in the expected JSON structure

2. **Convert INDRA Output**

   ```bash
   python indra_to_bel.py path/to/indra_reach_results.json
   ```

3. **Run Comparison**

   ```bash
   export OPENAI_API_KEY="your_api_key"
   python indra_vs_llm.py
   ```

   This will:
   - Parse and normalize both datasets
   - Find best matches for each LLM statement
   - Identify orphaned INDRA statements
   - Get LLM similarity ratings for paired statements
   - Generate `bel_comparison_results.csv`

4. **Analyze Results**

   ```bash
   jupyter notebook bel-analysis-notebook.ipynb
   ```

## Key Metrics

### Match Types

- **Best Match**: LLM statement paired with most similar INDRA statement
- **LLM Only**: Statements found only in TextToKG output
- **INDRA Only**: Statements found only in INDRA output

### Match Scores

- **1.0**: Perfect match across all components
- **0.7-0.9**: High similarity with minor differences
- **0.4-0.6**: Moderate similarity, same relationship but different entities
- **<0.4**: Low similarity, likely different biological facts

### Similarity Ratings (LLM-based)

- **Good**: Same or very similar biological relationships
- **Medium**: Related but with minor differences in specificity
- **Bad**: Different biological relationships or major conflicts
- **INDRA Only**: No corresponding LLM statement

## Customization

### Modifying File Paths

In `indra_vs_llm.py`, update the file paths in the `main()` function:

```python
llm_data, indra_data = comparator.load_data(
    'path/to/your/texttoKG_cleaned.json',
    'path/to/your/indra_bel_cleaned.json'
)
```

### Adding New Modification Mappings

In `BELStatementParser.normalize_bel_modifications()`, add new patterns:

```python
modification_mappings = {
    # Existing mappings...
    r'your_pattern': 'YourAbbreviation',
}
```

### Adjusting Match Score Weights

In `BELComparator.calculate_match_score()`, modify the scoring weights:

```python
# Current weights:
# Relationship: 0.4
# Subject: 0.25  
# Object: 0.25
# String similarity: 0.1
```

## Notes

- The comparison uses GPT-4 for semantic similarity assessment, which incurs API costs
- Rate limiting is implemented (1 second delay between API calls)
- The parser handles various BEL formats and normalizes them for fair comparison
- Namespace matching is fuzzy to account for different identifier systems

## Future Improvements

- Command-line arguments for file paths
- Batch processing for LLM similarity ratings
- Additional visualization options
- Export of matched pairs for manual review
- Performance metrics stratified by relationship type
