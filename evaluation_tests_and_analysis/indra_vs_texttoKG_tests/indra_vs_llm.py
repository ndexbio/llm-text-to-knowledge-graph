import json
import re
import pandas as pd
from typing import Dict, List, Tuple
import openai
from dotenv import load_dotenv
import os
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class BELStatementParser:
    """Parse BEL statements into components"""

    def __init__(self):
        # Common BEL relationships
        self.relationships = [
            'directlyIncreases', 'directlyDecreases', 'increases', 'decreases',
            'causesNoChange', 'cnc', 'association', 'isA', 'partOf', 'hasComponent',
            'positiveCorrelation', 'pos', 'negativeCorrelation', 'neg', 'regulates', 'reg'
        ]
        # Sort by length (longest first) to avoid partial matches
        self.relationships.sort(key=len, reverse=True)

    def parse_bel_statement(self, statement: str) -> Dict:
        """Parse a BEL statement into subject, relationship, and object"""
        statement = statement.strip()

        # Normalize INDRA format identifiers first to remove '!' and convert to LLM format
        statement = self.normalize_indra_identifiers(statement)

        # Find the relationship
        relationship = None
        rel_start = -1
        rel_end = -1

        for rel in self.relationships:
            pattern = r'\b' + re.escape(rel) + r'\b'
            match = re.search(pattern, statement)
            if match:
                relationship = rel
                rel_start = match.start()
                rel_end = match.end()
                break

        if not relationship:
            # If no relationship found, treat entire statement as subject
            return {
                'subject': statement,
                'relationship': None,
                'object': None,
                'subject_namespaces': self.extract_namespaces(statement),
                'object_namespaces': []
            }

        # Extract subject and object
        subject = statement[:rel_start].strip()
        object_part = statement[rel_end:].strip()

        return {
            'subject': subject,
            'relationship': relationship,
            'object': object_part,
            'subject_namespaces': self.extract_namespaces(subject),
            'object_namespaces': self.extract_namespaces(object_part)
        }

    def extract_namespaces(self, text: str) -> List[str]:
        """Extract namespaces from a BEL component"""
        # Pattern to match namespace:identifier format (handles both INDRA and LLM formats)
        namespace_pattern = r'\b([A-Z][A-Z0-9]*):(?:[^,\s\)!]+|\".+?\")'
        matches = re.findall(namespace_pattern, text)
        return list(set(matches))  # Remove duplicates

    def normalize_indra_identifiers(self, text: str) -> str:
        """Convert INDRA format 'HGNC:391 ! AKT1' to LLM format 'HGNC:AKT1'"""
        # Pattern to match namespace:id ! name format
        indra_pattern = r'\b([A-Z][A-Z0-9]*):[\w\d]+ ! ([\w\d]+)'

        def replace_func(match):
            namespace = match.group(1)
            name = match.group(2)
            return f"{namespace}:{name}"

        return re.sub(indra_pattern, replace_func, text)

    def normalize_bel_modifications(self, text: str) -> str:
        """Normalize BEL modification terms to standard abbreviations"""
        # Define mappings from verbose terms to BEL standard abbreviations
        modification_mappings = {
            # Phosphorylation variations
            r'go:0006468 ! "protein phosphorylation"': 'Ph',
            r'go:0006468 ! \"protein phosphorylation\"': 'Ph',
            r'"protein phosphorylation"': 'Ph',
            r'\"protein phosphorylation\"': 'Ph',
            r'phosphorylation': 'Ph',

            # Acetylation variations
            r'go:0006473 ! "protein acetylation"': 'Ac',
            r'go:0006473 ! \"protein acetylation\"': 'Ac',
            r'"protein acetylation"': 'Ac',
            r'\"protein acetylation\"': 'Ac',
            r'acetylation': 'Ac',

            # Methylation variations
            r'go:0006479 ! "protein methylation"': 'Me',
            r'go:0006479 ! \"protein methylation\"': 'Me',
            r'"protein methylation"': 'Me',
            r'\"protein methylation\"': 'Me',
            r'methylation': 'Me',

            # Ubiquitination variations
            r'go:0016567 ! "protein ubiquitination"': 'Ub',
            r'go:0016567 ! \"protein ubiquitination\"': 'Ub',
            r'"protein ubiquitination"': 'Ub',
            r'\"protein ubiquitination\"': 'Ub',
            r'ubiquitination': 'Ub',
            r'ubiquitylation': 'Ub',

            # SUMOylation variations
            r'go:0016925 ! "protein sumoylation"': 'Sumo',
            r'go:0016925 ! \"protein sumoylation\"': 'Sumo',
            r'"protein sumoylation"': 'Sumo',
            r'\"protein sumoylation\"': 'Sumo',
            r'sumoylation': 'Sumo',

            # Deacetylation variations
            r'go:0006476 ! "protein deacetylation"': 'Deac',
            r'go:0006476 ! \"protein deacetylation\"': 'Deac',
            r'"protein deacetylation"': 'Deac',
            r'\"protein deacetylation\"': 'Deac',
            r'deacetylation': 'Deac',

            # Add more as needed...
        }

        normalized_text = text
        for pattern, replacement in modification_mappings.items():
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)

        return normalized_text


class BELComparator:
    """Compare BEL statements from the two sources"""

    def __init__(self):
        self.parser = BELStatementParser()

    def load_data(self, llm_file: str, indra_file: str) -> Tuple[List, List]:
        """Load data from JSON files"""
        with open(llm_file, 'r') as f:
            llm_data = json.load(f)

        with open(indra_file, 'r') as f:
            indra_data = json.load(f)

        return llm_data, indra_data

    def normalize_llm_data(self, llm_data: List) -> Dict:
        """Normalize LLM data structure to match comparison needs"""
        normalized = {}

        for entry in llm_data:
            index = entry['Index']
            if index not in normalized:
                normalized[index] = {}

            for result in entry['Result']:
                evidence = result['evidence']
                if evidence not in normalized[index]:
                    normalized[index][evidence] = []

                normalized[index][evidence].append({
                    'bel_statement': result['bel_statement']
                })

        return normalized

    def normalize_indra_data(self, indra_data: List) -> Dict:
        """Normalize INDRA data structure"""
        normalized = {}

        for entry in indra_data:
            index = entry['Index']
            if index not in normalized:
                normalized[index] = {}

            for evidence_entry in entry['evidences']:
                evidence = evidence_entry['Evidence']
                if evidence not in normalized[index]:
                    normalized[index][evidence] = []

                for result in evidence_entry['Results']:
                    normalized[index][evidence].append({
                        'bel_statement': result['bel_statement']
                    })

        return normalized

    def are_components_semantically_equal(self, comp1: str, comp2: str) -> bool:
        """Check if two BEL statements are semantically equivalent after normalization"""
        if not comp1 or not comp2:
            return comp1 == comp2

        # Normalize both components
        norm1 = self.parser.normalize_bel_modifications(self.parser.normalize_indra_identifiers(comp1))
        norm2 = self.parser.normalize_bel_modifications(self.parser.normalize_indra_identifiers(comp2))

        # Remove extra whitespace and compare
        return norm1.strip() == norm2.strip()

    def calculate_match_score(self, llm_parsed: Dict, indra_parsed: Dict) -> float:
        """Calculate a match score between two parsed BEL statements"""
        score = 0.0

        # Relationship match (highest weight)
        if llm_parsed['relationship'] == indra_parsed['relationship']:
            score += 0.4

        # Subject semantic equivalence
        if self.are_components_semantically_equal(llm_parsed['subject'], indra_parsed['subject']):
            score += 0.25
        else:
            # Fallback to namespace overlap if not exactly equal
            llm_subj_ns = set(llm_parsed['subject_namespaces'])
            indra_subj_ns = set(indra_parsed['subject_namespaces'])
            if llm_subj_ns and indra_subj_ns:
                subj_overlap = len(llm_subj_ns.intersection(indra_subj_ns)) / max(len(llm_subj_ns), len(indra_subj_ns))
                score += 0.15 * subj_overlap
        # Object semantic equivalence
        if self.are_components_semantically_equal(llm_parsed['object'], indra_parsed['object']):
            score += 0.25
        else:
            # Fallback to namespace overlap if not exactly equal
            llm_obj_ns = set(llm_parsed['object_namespaces'])
            indra_obj_ns = set(indra_parsed['object_namespaces'])
            if llm_obj_ns and indra_obj_ns:
                obj_overlap = len(llm_obj_ns.intersection(indra_obj_ns)) / max(len(llm_obj_ns), len(indra_obj_ns))
                score += 0.15 * obj_overlap

        # String similarity as final tiebreaker (lower weight)
        if llm_parsed['subject'] and indra_parsed['subject']:
            subj_sim = self.simple_string_similarity(llm_parsed['subject'], indra_parsed['subject'])
            score += 0.05 * subj_sim

        if llm_parsed['object'] and indra_parsed['object']:
            obj_sim = self.simple_string_similarity(llm_parsed['object'], indra_parsed['object'])
            score += 0.05 * obj_sim

        return score

    def simple_string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity based on common tokens"""
        tokens1 = set(re.findall(r'\w+', s1.lower()))
        tokens2 = set(re.findall(r'\w+', s2.lower()))

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def find_best_matches(self, llm_statements: List, indra_statements: List) -> List[Tuple]:
        """Find best matches between LLM and INDRA statements"""
        matches = []

        for llm_stmt in llm_statements:
            llm_parsed = self.parser.parse_bel_statement(llm_stmt['bel_statement'])

            best_match = None
            best_score = 0.0

            for indra_stmt in indra_statements:
                indra_parsed = self.parser.parse_bel_statement(indra_stmt['bel_statement'])
                score = self.calculate_match_score(llm_parsed, indra_parsed)

                if score > best_score:
                    best_score = score
                    best_match = (llm_stmt, indra_stmt, llm_parsed, indra_parsed, score)

            if best_match:
                matches.append(best_match)
            else:
                # No match found
                matches.append((llm_stmt, None, llm_parsed, None, 0.0))

        return matches

    def create_best_match_plus_singles(self, llm_data: Dict, indra_data: Dict) -> List[Dict]:
        """Create best matches plus single INDRA statements which have no LLM match"""
        comparison_results = []
        used_indra_statements = set()  # Track which INDRA statements have been matched

        # Step 1: Find best matches for each LLM statement
        for index in llm_data:
            for evidence in llm_data[index]:
                llm_statements = llm_data[index][evidence]
                indra_statements = indra_data.get(index, {}).get(evidence, [])

                # Find best matches (original algorithm)
                matches = self.find_best_matches(llm_statements, indra_statements)

                for match in matches:
                    llm_stmt, indra_stmt, llm_parsed, indra_parsed, score = match

                    # Track which INDRA statement was used
                    if indra_stmt:
                        used_indra_statements.add((index, evidence, indra_stmt['bel_statement']))

                    comparison_results.append({
                        'index': index,
                        'evidence': evidence,
                        'llm_statement': llm_stmt['bel_statement'],
                        'indra_statement': indra_stmt['bel_statement'] if indra_stmt else None,
                        'llm_subject': llm_parsed['subject'],
                        'llm_relationship': llm_parsed['relationship'],
                        'llm_object': llm_parsed['object'],
                        'indra_subject': indra_parsed['subject'] if indra_parsed else None,
                        'indra_relationship': indra_parsed['relationship'] if indra_parsed else None,
                        'indra_object': indra_parsed['object'] if indra_parsed else None,
                        'subject_match': (
                            self.are_components_semantically_equal(
                                llm_parsed['subject'],
                                indra_parsed['subject']
                            ) if indra_parsed else False
                        ),
                        'relationship_match': (
                            llm_parsed['relationship'] == indra_parsed['relationship']
                            if indra_parsed else False
                        ),
                        'object_match': (
                            self.are_components_semantically_equal(
                                llm_parsed['object'],
                                indra_parsed['object']
                            ) if indra_parsed else False
                        ),
                        'subject_namespace_match': (
                            bool(
                                set(llm_parsed['subject_namespaces']).intersection(
                                    set(indra_parsed['subject_namespaces'])
                                )
                            ) if indra_parsed else False
                        ),
                        'object_namespace_match': (
                            bool(
                                set(llm_parsed['object_namespaces']).intersection(
                                    set(indra_parsed['object_namespaces'])
                                )
                            ) if indra_parsed else False
                        ),
                        'match_score': score,
                        'similarity_rating': None,  # To be filled by LLM for all pairs
                        'match_type': 'Best Match' if indra_stmt else 'LLM Only'
                    })

        # Step 2: Add orphaned INDRA statements (those not matched to any LLM statement)
        for index in indra_data:
            for evidence in indra_data[index]:
                indra_statements = indra_data[index][evidence]
                for indra_stmt in indra_statements:
                    # Check if this INDRA statement was already matched
                    if (index, evidence, indra_stmt['bel_statement']) not in used_indra_statements:
                        indra_parsed = self.parser.parse_bel_statement(indra_stmt['bel_statement']) 
                        comparison_results.append({
                            'index': index,
                            'evidence': evidence,
                            'llm_statement': None,
                            'indra_statement': indra_stmt['bel_statement'],
                            'llm_subject': None,
                            'llm_relationship': None,
                            'llm_object': None,
                            'indra_subject': indra_parsed['subject'],
                            'indra_relationship': indra_parsed['relationship'],
                            'indra_object': indra_parsed['object'],
                            'subject_match': False,
                            'relationship_match': False,
                            'object_match': False,
                            'subject_namespace_match': False,
                            'object_namespace_match': False,
                            'match_score': 0.0,
                            'similarity_rating': "INDRA Only",
                            'match_type': 'INDRA Only'
                        })

        return comparison_results

    def get_llm_similarity_rating(self, llm_statement: str, indra_statement: str) -> str:
        """Use LLM to rate similarity between two BEL statements"""
        if not indra_statement:
            return "No Match"

        prompt = f"""
        You are a biological knowledge expert. Compare these two BEL (Biological Expression Language) statements 
        and rate their similarity:

        LLM Statement: {llm_statement}
        INDRA Statement: {indra_statement}

        Rate the similarity as:
        - "Good": The statements represent the same or very similar biological relationships
        - "Medium": The statements are related but have minor differences in specificity or representation
        - "Bad": The statements represent different biological relationships or have major conflicts

        Consider:
        - Biological equivalence (not just textual similarity)
        - Whether different representations might mean the same thing biologically
        - Namespace differences that might be equivalent (e.g., different identifiers for same entity)

        Respond with EXACTLY one word: Good, Medium, or Bad
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", 
                     "content": "You are a biological knowledge expert specializing in BEL statement analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )

            rating = response.choices[0].message.content.strip()

            # More flexible validation - handle common variations
            rating_lower = rating.lower().replace('.', '').replace(',', '')

            if rating_lower in ['good']:
                return "Good"
            elif rating_lower in ['medium', 'okay', 'ok']:
                return "Medium"
            elif rating_lower in ['bad', 'poor']:
                return "Bad"
            else:
                print(f"Unexpected LLM response: '{rating}' - treating as Unknown")
                return "Unknown"

        except Exception as e:
            print(f"Error getting LLM rating: {e}")
            return "Error"

    def add_similarity_ratings(self, comparison_data: List[Dict]) -> List[Dict]:
        """Add LLM similarity ratings for all statement pairs"""
        print("Getting LLM similarity ratings for all paired statements...")

        # Only those rows with both an INDRA _and_ an LLM statement
        paired_statements = [
            row for row in comparison_data
            if row.get('indra_statement') and row.get('llm_statement')
        ]
        print(f"Found {len(paired_statements)} fully paired statements "
            f"(out of {len(comparison_data)} total)")

        for i, row in enumerate(paired_statements, 1):
            print(f"Processing {i}/{len(paired_statements)} (LLM rating needed)")
            rating = self.get_llm_similarity_rating(
                row['llm_statement'],
                row['indra_statement']
            )
            row['similarity_rating'] = rating
            time.sleep(1)  # Rate limiting

        # Rows without both an INDRA and LLM statement keep whatever similarity_rating they already have
        return comparison_data

    def save_results(self, comparison_data: List[Dict], output_file: str):
        """Save results to CSV and JSON"""
        # Save as CSV
        df = pd.DataFrame(comparison_data)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Results saved to {output_file}")

        # Print summary statistics
        matched_pairs = len([r for r in comparison_data if r['indra_statement']])
        print(f"Paired statements: {len([r for r in comparison_data if r['indra_statement'] is not None])}")
        print(f"Best matches: {len([r for r in comparison_data if r['match_type'] == 'Best Match'])}")
        print(f"LLM-only statements: {len([r for r in comparison_data if r['match_type'] == 'LLM Only'])}")
        print(f"INDRA-only statements: {len([r for r in comparison_data if r['match_type'] == 'INDRA Only'])}")

        if matched_pairs > 0:
            similarity_counts = {}
            for row in comparison_data:
                rating = row['similarity_rating']
                similarity_counts[rating] = similarity_counts.get(rating, 0) + 1

            print("\nSimilarity ratings:")
            for rating, count in similarity_counts.items():
                print(f"  {rating}: {count}")

        # Print match statistics
        if matched_pairs > 0:
            subject_matches = len([r for r in comparison_data if r['subject_match']])
            relationship_matches = len([r for r in comparison_data if r['relationship_match']])
            object_matches = len([r for r in comparison_data if r['object_match']])

            print(f"\nComponent matches (out of {matched_pairs} pairs):")
            print(f"  Subject matches: {subject_matches} ({subject_matches/matched_pairs*100:.1f}%)")
            print(f"  Relationship matches: {relationship_matches} ({relationship_matches/matched_pairs*100:.1f}%)")
            print(f"  Object matches: {object_matches} ({object_matches/matched_pairs*100:.1f}%)")


def main():
    # Initialize comparator
    comparator = BELComparator()

    # Load data
    print("Loading data...")
    llm_data, indra_data = comparator.load_data('evaluation_tests_and_analysis/indra_vs_texttoKG_tests/indra_bel_cleaned.json',
                                                'evaluation_tests_and_analysis/indra_vs_texttoKG_tests/texttoKG_cleaned.json')

    # Normalize data structures
    print("Normalizing data structures...")
    llm_normalized = comparator.normalize_llm_data(llm_data)
    indra_normalized = comparator.normalize_indra_data(indra_data)

    # Create best matches plus orphaned INDRA statements
    print("Creating best matches plus orphaned statements...")
    comparison_data = comparator.create_best_match_plus_orphans(llm_normalized, indra_normalized)

    print(f"Found {len(comparison_data)} total comparisons")

    # Add LLM similarity ratings 
    comparison_data = comparator.add_similarity_ratings(comparison_data)

    # Save results
    comparator.save_results(comparison_data, 'evaluation_tests/indra_vs_texttoKG_tests/bel_comparison_results.json')

    print("Comparison complete!")


if __name__ == "__main__":
    main()
