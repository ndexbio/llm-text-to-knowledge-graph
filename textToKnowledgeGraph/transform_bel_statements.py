def parse_bel_statement(bel_statement):
    bel_statement = bel_statement.strip()

    depth = 0
    for ch in bel_statement:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ' ' and depth == 0:
            # found a relation separator → go parse binary below
            break
    else:
        # loop fell through (no top‐level space found) → unary
        return bel_statement, None, None

    def extract_expression(s, start):
        # skip whitespace…
        while start < len(s) and s[start].isspace():
            start += 1
        func_start = start
        # read until '(' or token break
        while start < len(s) and (s[start].isalnum() or s[start] in ['_', ':', '"']):
            # if we hit a quote, consume the whole quoted string
            if s[start] == '"':
                start += 1
                while start < len(s) and s[start] != '"':
                    start += 1
                start += 1
            else:
                start += 1

        # if there’s no '(', treat what we just read as a bare term
        if start >= len(s) or s[start] != '(':
            expr = s[func_start:start]
            return expr, start

        # otherwise it’s a function-call—fall back to your existing paren‐matching logic
        func_name = s[func_start:start]
        stack = []
        expr_start = start
        while start < len(s):
            if s[start] == '(':
                stack.append('(')
            elif s[start] == ')':
                stack.pop()
                if not stack:
                    start += 1
                    break
            start += 1
        expr = func_name + s[expr_start:start]
        return expr, start

    # Extract left (source) expression
    left_expr, idx = extract_expression(bel_statement, 0)
    if left_expr is None:
        return None, None, None

    # Skip whitespace to get the relation word
    while idx < len(bel_statement) and bel_statement[idx].isspace():
        idx += 1
    rel_start = idx

    # Stop at whitespace OR at the '(' that starts the right‐hand expression
    while idx < len(bel_statement) and not bel_statement[idx].isspace() and bel_statement[idx] != '(':
        idx += 1
    relation = bel_statement[rel_start:idx]

    # Skip whitespace and extract right (target) expression
    while idx < len(bel_statement) and bel_statement[idx].isspace():
        idx += 1
    right_expr, idx = extract_expression(bel_statement, idx)
    if right_expr is None:
        return None, None, None

    return left_expr, relation, right_expr


def process_llm_results(llm_data):
    extracted_results = []

    for entry in llm_data.get("LLM_extractions", []):
        text = entry.get("text", "")
        for result in entry.get("Results", []):
            bel_stmt = result.get("bel_statement", "")
            evidence = result.get("evidence", "")
            source, interaction, target = parse_bel_statement(bel_stmt)
            if source and interaction and target:
                extracted_results.append({
                    "source": source,
                    "interaction": interaction,
                    "target": target,
                    "text": text,
                    "evidence": evidence
                })

        for annots in entry["annotations"]:
            entry_name = annots["entry_name"]
            url = annots["url"]
            extracted_results.append({
                "entry_name": entry_name,
                "url": url
            }) 

    return extracted_results
