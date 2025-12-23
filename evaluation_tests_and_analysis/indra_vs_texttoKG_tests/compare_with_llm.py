"""Compare LLM and INDRA statements for semantic similarity using OpenAI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
# from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency issue should be explicit
    raise SystemExit(
        "The openai package is required. Install dependencies via `uv run --script compare_statements.py --help`."
    ) from exc


SCORE_TO_RATING = {0: "low", 2: "partial", 4: "good"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LLM statements against INDRA statements (JSON input) grouped by evidence."
        )
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="Path to the LLM JSON file (e.g. texttoKG_cleaned.json)",
    )
    parser.add_argument(
        "--indra",
        required=True,
        help="Path to the INDRA JSON file (e.g. indra_bel_cleaned.json)",
    )
    parser.add_argument(
        "--output",
        default="comparison_output.csv",
        help="Path to write the comparison JSON output",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI model to use for comparisons (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls (mark all comparisons as none_comparable).",
    )
    return parser.parse_args()


def sanitize(value: Optional[str]) -> str:
    return value.strip() if value else ""


def load_json_files(llm_path: str, indra_path: str):
    """Load and merge LLM + INDRA JSONs into comparable structures."""
    try:
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_data = json.load(f)
        with open(indra_path, "r", encoding="utf-8") as f:
            indra_data = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not find one of the JSON files: {exc}") from exc

    # Create mapping for INDRA entries by Index
    indra_map = {str(entry["Index"]): entry for entry in indra_data}
    merged_rows = []

    for llm_entry in llm_data:
        idx = str(llm_entry["Index"])
        llm_results = llm_entry.get("Result", [])
        indra_entry = indra_map.get(idx, {})
        evidences = indra_entry.get("evidences", [])

        for llm_result in llm_results:
            evidence = llm_result.get("evidence", "")
            llm_stmt = llm_result.get("bel_statement", "")
            # For matching INDRA evidence(s)
            if evidences:
                for ev_obj in evidences:
                    for indra_result in ev_obj.get("Results", []):
                        merged_rows.append({
                            "index": idx,
                            "evidence": evidence or ev_obj.get("Evidence", ""),
                            "llm_statement": llm_stmt,
                            "indra_statement": indra_result.get("bel_statement", "")
                        })
            else:
                # No INDRA match found
                merged_rows.append({
                    "index": idx,
                    "evidence": evidence,
                    "llm_statement": llm_stmt,
                    "indra_statement": ""
                })
    return merged_rows


def extract_json_object(text: str) -> Dict[str, object]:
    """Extract the first JSON object present in *text*."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in response")
    fragment = text[start: end + 1]
    return json.loads(fragment)


def call_openai(
    client: OpenAI,
    model: str,
    llm_statement: str,
    indra_statements: List[str],
) -> List[Dict[str, object]]:
    """
    Compare one LLM statement against all INDRA statements.
    Returns a list of dicts containing semantic and sentence meaning similarity scores.
    """

    numbered_candidates = "\n".join(
        f"{idx}. {candidate}" for idx, candidate in enumerate(indra_statements)
    )

    user_prompt = (
        "You will receive one query statement followed by numbered candidate statements. "
        "For each candidate, evaluate two aspects:\n"
        "1. **Semantic similarity** — check whether the BEL structures share the same biological entities "
        "and relationship (e.g., same genes/proteins and causal direction).\n"
        "2. **Biological similarity** — check whether both statements describe the same "
        "biological interaction or event, even if represented differently.\n\n"
        "Rate each candidate for both dimensions using:\n"
        "0 = low, 2 = partial, 4 = good, or 'none' if not comparable.\n\n"
        "Respect the following mapping strictly:\n"
        "0 -> low similarity, 2 -> partial similarity, 4 -> good similarity.\n"
        "If a statement is not comparable, use \"semantic\": \"none\" and \"meaning\": \"none\".\n\n"
        "Each entry must include an \"explanation\" (1–2 sentences) describing why those ratings were assigned.\n\n"
        "Return only valid JSON in this exact format:\n"
        "[\n"
        "  {\"index\": <candidate_index>, "
        "   \"semantic\": <0|2|4|\"none\">, "
        "   \"meaning\": <0|2|4|\"none\">, "
        "   \"explanation\": \"<brief reason>\"}, ...\n"
        "]\n\n"
        "Query statement:\n"
        f"{llm_statement}\n\n"
        "Candidate statements:\n"
        f"{numbered_candidates}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in biological knowledge representation. "
                           "Respond strictly with valid JSON as specified, no commentary."
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    message = response.choices[0].message.content or ""

    # Try to extract JSON list safely
    start = message.find("[")
    end = message.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON list found in model response.")
    parsed = json.loads(message[start:end + 1])

    # Validate and normalize structure
    results = []
    for item in parsed:
        try:
            idx = int(item["index"])
            sem = item.get("semantic")
            mean = item.get("meaning")
            expl = str(item.get("explanation", "")).strip()

            # Normalize numeric fields
            sem_score = None if sem in ("none", None) else int(sem)
            mean_score = None if mean in ("none", None) else int(mean)

            results.append({
                "match_index": idx,
                "semantic_score": sem_score,
                "similarity_score": mean_score,
                "explanation": expl,
            })
        except Exception as e:
            print(f"[warning] Skipped invalid item in response: {item} ({e})")
    return results


def group_rows(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    """Organize rows by evidence with helper structures for downstream processing."""
    grouped: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "llm_rows": defaultdict(list),
            "indra_candidates": [],
        }
    )

    for idx, row in enumerate(rows):
        evidence = sanitize(row.get("evidence"))
        llm_statement = sanitize(row.get("llm_statement"))
        indra_statement = sanitize(row.get("indra_statement"))

        bucket = grouped[evidence]
        if llm_statement:
            bucket["llm_rows"][llm_statement].append(idx)
        if indra_statement and indra_statement not in bucket["indra_candidates"]:
            bucket["indra_candidates"].append(indra_statement)

    return grouped


def main() -> None:
    args = parse_args()
    load_dotenv()

    rows = load_json_files(args.llm, args.indra)
    grouped = group_rows(rows)

    # Prepare result placeholders per row.
    augmented: List[Dict[str, str]] = [dict(row) for row in rows]
    row_annotations = [
        {
            "match_type": "not_compared",
            "similarity_rating_reason": "",
            "semantic_score": None,
            "similarity_score": None,
        }
        for _ in rows
    ]

    # Initial classification for rows lacking either statement.
    for idx, row in enumerate(rows):
        llm_statement = sanitize(row.get("llm_statement"))
        indra_statement = sanitize(row.get("indra_statement"))
        if llm_statement and not indra_statement:
            row_annotations[idx]["match_type"] = "llm_only"
            row_annotations[idx]["similarity_rating_reason"] = (
                "No INDRA statement available for comparison."
            )
        elif indra_statement and not llm_statement:
            row_annotations[idx]["match_type"] = "indra_only"
            row_annotations[idx]["similarity_rating_reason"] = (
                "No LLM statement available for comparison."
            )
        elif not llm_statement and not indra_statement:
            row_annotations[idx]["similarity_rating_reason"] = "Row has no statements to compare."

    client: Optional[OpenAI] = None
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=<your key> or use --dry-run."
            )
        client = OpenAI(api_key=api_key)

    for evidence, structures in grouped.items():
        indra_candidates: List[str] = structures["indra_candidates"]
        if not indra_candidates:
            # No INDRA statements to compare against for this evidence.
            for _, indices in structures["llm_rows"].items():
                for idx in indices:
                    row_annotations[idx]["match_type"] = "llm_only"
                    row_annotations[idx]["similarity_rating"] = "none_comparable"
                    row_annotations[idx]["similarity_rating_reason"] = (
                        "No INDRA statements available for this evidence."
                    )
            continue

        for llm_statement, indices in structures["llm_rows"].items():
            if args.dry_run or client is None:
                decision = [
                    {
                        "match_index": i,
                        "semantic_score": None,
                        "similarity_score": None,
                        "explanation": "dry_run"
                    }
                    for i in range(len(indra_candidates))
                ]
            else:
                try:
                    decision = call_openai(client, args.model, llm_statement, indra_candidates)
                except Exception as exc:  # pragma: no cover - runtime protection
                    print(
                        f"[warning] Failed to obtain similarity for evidence '{evidence}': {exc}",
                        file=sys.stderr,
                    )
                    decision = []

            decision_map = {d["match_index"]: d for d in decision}
            if decision:
                best_decision = max(
                    decision,
                    key=lambda d: (d["similarity_score"] or 0)
                )
                best_sim = best_decision.get("similarity_score") or 0
                if best_sim > 0:
                    best_index = best_decision["match_index"]
                else:
                    best_index = None
            else:
                best_index = None

            for i, indra_stmt in enumerate(indra_candidates):
                for idx in indices:
                    row_indra = sanitize(rows[idx].get("indra_statement"))
                    if row_indra == indra_stmt:
                        d = decision_map.get(i)
                        if d:
                            sim_score = d["similarity_score"]
                            sem_score = d["semantic_score"]

                            row_annotations[idx]["match_index"] = i
                            row_annotations[idx]["similarity_rating_reason"] = d["explanation"]
                            row_annotations[idx]["semantic_score"] = sem_score
                            row_annotations[idx]["similarity_score"] = sim_score

                            # Add human-readable ratings
                            row_annotations[idx]["semantic_rating"] = SCORE_TO_RATING.get(
                                sem_score, "none_comparable"
                            )
                            row_annotations[idx]["similarity_rating"] = SCORE_TO_RATING.get(
                                sim_score, "none_comparable"
                            )

                            # Assign match_type intelligently
                            if sim_score is None or sim_score == 0:
                                row_annotations[idx]["match_type"] = "not_similar"
                            elif i == best_index:
                                row_annotations[idx]["match_type"] = "most_similar"
                            else:
                                row_annotations[idx]["match_type"] = "rated"

                        else:
                            # No decision data available for this candidate
                            row_annotations[idx]["match_index"] = i
                            row_annotations[idx]["match_type"] = "not_comparable"
                            row_annotations[idx]["similarity_rating_reason"] = "No comparison available."
                            row_annotations[idx]["semantic_score"] = None
                            row_annotations[idx]["similarity_score"] = None

    fieldnames = list(rows[0].keys()) + [
        "match_index",
        "semantic_score",
        "semantic_rating",
        "similarity_score",
        "similarity_rating",
        "match_type",
        "similarity_rating_reason",
    ]

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, annotation in zip(augmented, row_annotations):
                row.update(annotation)
                writer.writerow(row)
    except OSError as exc:
        raise SystemExit(f"Failed to write output CSV: {exc}") from exc

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
