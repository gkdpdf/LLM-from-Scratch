import os
import re
from openai import OpenAI
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any,Optional
from dotenv import load_dotenv
load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Helpers ----------
def normalize(t):
    return re.sub(r'[^a-zA-Z0-9 ]+', '', str(t)).lower().strip()

def detect_time_filters(user_query: str):
    query = user_query.lower()
    today = datetime.today().date()
    if "last 2 months" in query:
        return {"time_range": [str(today - timedelta(days=60)), str(today)]}
    if "last 3 months" in query:
        return {"time_range": [str(today - timedelta(days=90)), str(today)]}
    if "last month" in query:
        return {"time_range": [str(today - timedelta(days=30)), str(today)]}
    if "last week" in query:
        return {"time_range": [str(today - timedelta(days=7)), str(today)]}
    return {}

def shortlist_candidates_with_scores(text, options, k=15, score_cutoff=60):
    if not options:
        return []
    text_norm = normalize(text)
    normalized_options = [normalize(o) for o in options]
    norm_to_original = dict(zip(normalized_options, options))

    exact_matches = []
    for norm_option, orig_option in norm_to_original.items():
        if text_norm in norm_option:
            score = 90 + (len(text_norm) / len(norm_option)) * 10
            exact_matches.append((orig_option, score))

    fuzzy_matches = process.extract(
        text_norm, normalized_options, scorer=fuzz.token_set_ratio,
        limit=k, score_cutoff=score_cutoff
    )

    all_matches = {}
    for match, score in exact_matches:
        all_matches[match] = score
    for match_result in fuzzy_matches:
        match, score = match_result[0], match_result[1]
        orig_match = norm_to_original[match]
        if orig_match not in all_matches:
            all_matches[orig_match] = score

    final_matches = list(all_matches.items())
    final_matches.sort(key=lambda x: x[1], reverse=True)
    return final_matches[:k]

# ---------- LLM Step 1 ----------
def llm_understand(user_query):
    prompt = f"""
You are a business query analyzer.

1. If the query is irrelevant to sales, products, distributors, or superstockists,
   return JSON with "intent": "irrelevant".

2. If relevant, extract:
   - intent (query|aggregation|ranking|comparison)
   - metrics (sales, revenue, quantity, etc.)
   - entities: Extract COMPLETE entity names, don't split them.

Examples:
- "VH trading" → {{"distributor": ["VH trading"]}}
- "takatak" → {{"product": ["takatak"]}}
- "Samsung Galaxy S21" → {{"product": ["Samsung Galaxy S21"]}}

User query:
```{user_query}```

Return JSON only with complete entity names.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Return only JSON with complete entity names."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return eval(resp.choices[0].message.content)
    except:
        return {"intent": "irrelevant", "metrics": [], "entities": {}}

# ---------- Entity Resolution ----------
def resolve_entity_with_disambiguation(entity_value, catalog, table_columns):
    """
    Resolve entity by:
    1. Checking all columns
    2. Asking user to pick column if multiple matches
    3. Asking user to pick value if multiple candidates
    """
    all_matches = {}
    for table, cols in catalog.items():
        for col, values in cols.items():
            candidates = shortlist_candidates_with_scores(entity_value, values)
            if candidates:
                all_matches[(table, col)] = candidates

    if not all_matches:
        return {"status": "not found", "value": entity_value}

    # Step 1: Let user pick column if entity appears in multiple columns
    if len(all_matches) > 1:
        print(f"\n🤔 '{entity_value}' found in multiple columns:")
        keys = list(all_matches.keys())
        for i, (table, col) in enumerate(keys, 1):
            best = all_matches[(table, col)][0][0]
            print(f"  {i}. {col} (in {table}) → best match: '{best}'")
        while True:
            try:
                choice = int(input(f"Which column do you mean? (1-{len(keys)}): "))
                if 1 <= choice <= len(keys):
                    selected_key = keys[choice - 1]
                    break
            except ValueError:
                pass
    else:
        selected_key = next(iter(all_matches.keys()))

    table, column = selected_key
    candidates = all_matches[selected_key]

    # Step 2: Let user pick entity if multiple values match in same column
    if len(candidates) > 1:
        print(f"\n🎯 Multiple matches found for '{entity_value}' in {column}:")
        for i, (cand, score) in enumerate(candidates, 1):
            print(f"  {i}. {cand} (similarity {score}%)")
        while True:
            try:
                choice = int(input(f"Which one do you mean? (1-{len(candidates)}): "))
                if 1 <= choice <= len(candidates):
                    final_value = candidates[choice - 1][0]
                    break
            except ValueError:
                pass
    else:
        final_value = candidates[0][0]

    return {"table": table, "column": column, "value": final_value}

# ---------- Main Resolver ----------
def resolve_with_human_in_loop_pg(user_query, catalog, table_columns):
    parsed = llm_understand(user_query)

    if parsed["intent"] == "irrelevant":
        print("🙅 This question doesn't relate to products, distributors, or sales in the DB.")
        return {"intent": "irrelevant"}

    intent = parsed.get("intent", "query")
    metrics = parsed.get("metrics", [])
    entities = parsed.get("entities", {})
    filters = detect_time_filters(user_query)

    resolved_entities = {}
    for entity_type, values in entities.items():
        if not values:
            continue
        print(f"\n🔍 Resolving entity: '{values[0]}'")
        result = resolve_entity_with_disambiguation(values[0], catalog, table_columns)
        if result.get("status") != "not found":
            resolved_entities[entity_type] = result

    # Ask user for table if not clear
    if "sales" in metrics and not any(tbl in user_query.lower() for tbl in ["primary", "shipment"]):
        print("\n❓ 'Sales' found. Do you mean:")
        print("  1. tbl_primary")
        print("  2. tbl_shipment")
        while True:
            try:
                choice = int(input("Select table (1-2): "))
                if choice in [1, 2]:
                    table = "tbl_primary" if choice == 1 else "tbl_shipment"
                    break
            except ValueError:
                pass
    else:
        if resolved_entities:
            table = next(iter(resolved_entities.values())).get("table")
        else:
            print("\n📊 Which table do you want to query?")
            for i, t in enumerate(table_columns.keys(), 1):
                print(f"  {i}. {t}")
            while True:
                try:
                    choice = int(input(f"Select table (1-{len(table_columns)}): "))
                    if 1 <= choice <= len(table_columns):
                        table = list(table_columns.keys())[choice-1]
                        break
                except ValueError:
                    pass

    candidate_cols = [ent["column"] for ent in resolved_entities.values() if "column" in ent]
    candidate_cols = list(dict.fromkeys(candidate_cols)) if candidate_cols else list(table_columns[table])

    print(f"\n📊 Candidate columns in {table}:")
    for i, col in enumerate(candidate_cols, 1):
        print(f"  {i}. {col}")
    cols_input = input("Select columns by number (comma separated, or Enter for auto): ")
    if cols_input.strip():
        col_indices = [int(x.strip()) for x in cols_input.split(",") if x.strip().isdigit()]
        selected_cols = [candidate_cols[i-1] for i in col_indices if 1 <= i <= len(candidate_cols)]
    else:
        selected_cols = candidate_cols

    return {
        "intent": intent,
        "metrics": metrics,
        "entities": resolved_entities,
        "filters": filters,
        "table": table,
        "columns": selected_cols
    }


import psycopg2

def build_catalog(conn, table_columns, max_values=50):
    """
    Build catalog = {table: {column: [distinct values...]}} 
    from Postgres DB.
    """
    catalog = {}
    cur = conn.cursor()

    for table, cols in table_columns.items():
        catalog[table] = {}
        for col in cols:
            try:
                q = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT {max_values};"
                cur.execute(q)
                values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                catalog[table][col] = values
            except Exception as e:
                print(f"⚠️ Skipping {table}.{col} → {e}")
    cur.close()
    return catalog


conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

import psycopg2

def load_table_columns_pg(conn, tables):
    """
    Load column names for given tables from PostgreSQL.
    Returns a dict {table_name: [col1, col2, ...]}
    """
    table_columns = {}
    with conn.cursor() as cur:
        for table in tables:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            cols = [row[0] for row in cur.fetchall()]
            table_columns[table] = cols
    return table_columns


# ---------- Example usage ----------
conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

# load multiple tables
tables_to_load = ["tbl_shipment", "tbl_primary", "tbl_product_master"]
table_columns = load_table_columns_pg(conn, tables_to_load)

# print("\n📊 Table Columns Loaded:")
# for tbl, cols in table_columns.items():
#     print(f"{tbl}: {cols}")

table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary","tbl_product_master"])
# print(table_columns)

# Get table + column structure
table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])

# Build the catalog with actual values
catalog = build_catalog(conn, table_columns)



class GraphState(TypedDict, total=False):
    user_query: str
    catalog: Dict[str, Any]              
    table_columns: Dict[str, List[str]]  
    annotated_schema: str
    resolved: Dict[str, Any]
    sql_result: Any                      
    validated_sql: str                  
    validation_status: str               
    validation_error: Optional[str]
    execution_result: Any                
    execution_status: str               
    execution_error: Optional[str]
    route_decision: str                
    final_output: str                    
    reasoning_trace: List[str]


def entity_resolver_node(state: GraphState):
    """
    Resolves entities (products, distributors, etc.) FIRST,
    then provides annotated schema.
    """
    user_query = state["user_query"]
    catalog = state.get("catalog", {})
    table_columns = state.get("table_columns", {})

    reasoning = []

    # --- Try to resolve entities ---
    resolved = resolve_with_human_in_loop_pg(user_query, catalog, table_columns)

    # Safety: If resolver failed, fallback to raw query text
    if not resolved.get("entities"):
        reasoning.append(f"⚠️ No entity found in schema for '{user_query}'. Passing raw text downstream.")
        resolved = {
            "intent": "fallback",
            "entities": {"raw_text": user_query},
            "filters": {},
            "message": "Entity not found, using raw query."
        }

    # --- Annotated Schema (from file) ---
    try:
        with open("annotated_schema.md", "r", encoding="utf-8") as f:
            annotated_schema = f.read()
    except FileNotFoundError:
        annotated_schema = """
        ### tbl_primary
        - product_id → references tbl_product_master.product_erp_id
        - distributor_name : name of distributor
        - sales_order_date : date of order
        - invoiced_total_quantity : actual billed sales

        ### tbl_product_master
        - product_erp_id : unique product key
        - product : product description/name
        """

    resolved["message"] = "Entities resolved or fallback applied."
    resolved["thinking"] = reasoning

    return {
        "resolved": resolved,
        "annotated_schema": annotated_schema
    }
