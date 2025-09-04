from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any,Optional
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import sqlparse
import psycopg2
from dotenv import load_dotenv
load_dotenv()

# Simple LLM executors (no agents needed)
class SQLAgentExecutor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=os.getenv('OPENAI_API_KEY'))
    
    def invoke(self, inputs):
        input_text = inputs.get("input", "")
        result = self.llm.invoke(f"""You are a PostgreSQL query generator. 
Generate only valid PostgreSQL queries based on the provided schema and user requirements.
Do not create new tables or columns. Use only what exists in the schema.
Return only the SQL query without explanations.

{input_text}""")
        return {"output": result.content}

class ValidationAgentExecutor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=os.getenv('OPENAI_API_KEY'))
    
    def invoke(self, inputs):
        input_text = inputs.get("input", "")
        result = self.llm.invoke(f"""You are a PostgreSQL query validator. 
Validate SQL queries against database schemas.
Respond with 'VALID' if correct, or 'INVALID: [specific error]' if not.

{input_text}""")
        return {"output": result.content}

# Create executors
sql_agent_executor = SQLAgentExecutor()
validation_agent_executor = ValidationAgentExecutor()


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


conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

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

print("\n📊 Table Columns Loaded:")
for tbl, cols in table_columns.items():
    print(f"{tbl}: {cols}")

table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary","tbl_product_master"])
print(table_columns)

# Get table + column structure
table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])

# Build the catalog with actual values
catalog = build_catalog(conn, table_columns)
import sqlparse
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI

import sqlparse
import re
import psycopg2
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
import os

class DatabaseValidator:
    def __init__(self, connection_params):
        self.connection_params = connection_params
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    def get_actual_table_structure(self):
        """Get real table structure from database"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor()
            
            table_structure = {}
            
            # Get all tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get columns for each table
            for table in tables:
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (table,))
                
                columns = cursor.fetchall()
                table_structure[table] = {
                    'columns': [col[0] for col in columns],
                    'types': {col[0]: col[1] for col in columns}
                }
            
            cursor.close()
            conn.close()
            
            return table_structure
            
        except Exception as e:
            print(f"⚠️ Could not get table structure: {e}")
            return {}

def validator_agent(state: GraphState):
    """
    Complete validator that handles all errors in background and auto-corrects
    """
    sql_query = state.get('sql_result', '')
    annotated_schema = state.get('annotated_schema', '')
    user_query = state.get('user_query', '')
    
    print(f"🔍 VALIDATOR: Starting validation for SQL: {sql_query}")
    
    # Connection params for validation
    connection_params = {
        'host': 'localhost',
        'database': 'haldiram',
        'user': 'postgres',
        'password': '12345678',
        'port': 5432
    }
    
    validator = DatabaseValidator(connection_params)
    
    # Step 1: Get real database structure
    print("📊 Getting actual database structure...")
    table_structure = validator.get_actual_table_structure()
    
    # Step 2: Multi-level validation and correction
    max_attempts = 5
    current_sql = sql_query
    
    for attempt in range(max_attempts):
        print(f"🔄 Validation attempt {attempt + 1}/{max_attempts}")
        print(f"   Current SQL: {current_sql}")
        
        # Level 1: Basic SQL syntax validation
        syntax_valid, syntax_error = validate_sql_syntax(current_sql)
        if not syntax_valid:
            print(f"❌ Syntax error: {syntax_error}")
            current_sql = fix_syntax_error(current_sql, syntax_error, validator.llm)
            continue
        
        # Level 2: Table and column existence validation
        structure_valid, structure_error = validate_table_structure(current_sql, table_structure)
        if not structure_valid:
            print(f"❌ Structure error: {structure_error}")
            current_sql = fix_structure_error(current_sql, structure_error, table_structure, validator.llm)
            continue
        
        # Level 3: Database execution test
        execution_valid, execution_error = test_sql_execution(current_sql, connection_params)
        if not execution_valid:
            print(f"❌ Execution error: {execution_error}")
            current_sql = fix_execution_error(current_sql, execution_error, table_structure, validator.llm)
            continue
        
        # Level 4: Logic validation with LLM
        logic_valid, logic_error = validate_sql_logic(current_sql, user_query, table_structure, validator.llm)
        if not logic_valid:
            print(f"❌ Logic error: {logic_error}")
            current_sql = fix_logic_error(current_sql, logic_error, user_query, table_structure, validator.llm)
            continue
        
        # All validations passed!
        print(f"✅ SQL validated and corrected successfully!")
        return {
            "validated_sql": current_sql,
            "validation_status": "valid" if attempt == 0 else "corrected",
            "validation_error": None,
            "correction_attempts": attempt
        }
    
    # Max attempts reached - return best effort
    print(f"⚠️ Max validation attempts reached. Using last version.")
    return {
        "validated_sql": current_sql,
        "validation_status": "max_attempts_reached",
        "validation_error": f"Could not fully validate after {max_attempts} attempts",
        "correction_attempts": max_attempts
    }


def validate_sql_syntax(sql_query: str) -> tuple[bool, str]:
    """Validate basic SQL syntax"""
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed or len(parsed) == 0:
            return False, "Invalid SQL syntax - no valid statements found"
        
        statement = parsed[0]
        if not statement.tokens:
            return False, "Empty SQL statement"
        
        return True, ""
        
    except Exception as e:
        return False, f"SQL parsing error: {str(e)}"


def validate_table_structure(sql_query: str, table_structure: Dict) -> tuple[bool, str]:
    """Validate that tables and columns exist"""
    try:
        # Extract table names from SQL
        tables_in_query = extract_table_names(sql_query)
        
        # Check if tables exist
        for table in tables_in_query:
            if table not in table_structure:
                return False, f"Table '{table}' does not exist. Available tables: {list(table_structure.keys())}"
        
        # Extract column references and validate
        column_refs = extract_column_references(sql_query)
        
        for table_alias, column in column_refs:
            # Find actual table name for alias
            actual_table = find_table_for_alias(sql_query, table_alias, table_structure)
            
            if actual_table and column not in table_structure[actual_table]['columns']:
                available_cols = table_structure[actual_table]['columns']
                return False, f"Column '{column}' does not exist in table '{actual_table}'. Available columns: {available_cols}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Structure validation error: {str(e)}"


def test_sql_execution(sql_query: str, connection_params: Dict) -> tuple[bool, str]:
    """Test SQL execution without committing"""
    try:
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()
        
        # Use EXPLAIN to test query without executing
        explain_query = f"EXPLAIN {sql_query}"
        cursor.execute(explain_query)
        
        cursor.close()
        conn.close()
        
        return True, ""
        
    except psycopg2.Error as e:
        error_code = e.pgcode if hasattr(e, 'pgcode') else 'Unknown'
        error_msg = str(e).split('\n')[0]  # Get first line of error
        return False, f"PostgreSQL Error {error_code}: {error_msg}"
    except Exception as e:
        return False, f"Execution test error: {str(e)}"


def validate_sql_logic(sql_query: str, user_query: str, table_structure: Dict, llm) -> tuple[bool, str]:
    """Validate SQL logic using LLM"""
    try:
        # Create schema description for LLM
        schema_desc = create_schema_description(table_structure)
        
        validation_prompt = f"""
Validate if this SQL query correctly answers the user question:

User Question: {user_query}
SQL Query: {sql_query}
Database Schema: {schema_desc}

Check if:
1. The query logic matches the user's intent
2. Proper joins are used
3. Filters are appropriate
4. The result would answer the question

Respond with either:
"VALID" - if the query correctly answers the question
"INVALID: [specific issue]" - if there are logic problems
"""
        
        result = llm.invoke(validation_prompt)
        response = result.content.strip()
        
        if response.startswith("VALID"):
            return True, ""
        else:
            error = response.replace("INVALID:", "").strip()
            return False, error
            
    except Exception as e:
        return True, ""  # Skip logic validation if LLM fails


# Helper functions for fixing errors

def fix_syntax_error(sql_query: str, error: str, llm) -> str:
    """Fix basic syntax errors"""
    correction_prompt = f"""
Fix the syntax error in this SQL query:

SQL: {sql_query}
Error: {error}

Return only the corrected SQL query, no explanations.
"""
    
    try:
        result = llm.invoke(correction_prompt)
        corrected = clean_sql_output(result.content)
        return corrected if corrected else sql_query
    except:
        return sql_query


def fix_structure_error(sql_query: str, error: str, table_structure: Dict, llm) -> str:
    """Fix table/column structure errors"""
    schema_desc = create_schema_description(table_structure)
    
    correction_prompt = f"""
Fix the table/column error in this SQL query:

SQL: {sql_query}
Error: {error}
Correct Database Schema: {schema_desc}

Fix the column/table names to match the actual schema.
Return only the corrected SQL query.
"""
    
    try:
        result = llm.invoke(correction_prompt)
        corrected = clean_sql_output(result.content)
        return corrected if corrected else sql_query
    except:
        return sql_query


def fix_execution_error(sql_query: str, error: str, table_structure: Dict, llm) -> str:
    """Fix SQL execution errors"""
    schema_desc = create_schema_description(table_structure)
    
    correction_prompt = f"""
Fix this PostgreSQL execution error:

SQL: {sql_query}
Execution Error: {error}
Database Schema: {schema_desc}

Generate a corrected SQL query that will execute successfully.
Return only the SQL query.
"""
    
    try:
        result = llm.invoke(correction_prompt)
        corrected = clean_sql_output(result.content)
        return corrected if corrected else sql_query
    except:
        return sql_query


def fix_logic_error(sql_query: str, error: str, user_query: str, table_structure: Dict, llm) -> str:
    """Fix logical errors in SQL"""
    schema_desc = create_schema_description(table_structure)
    
    correction_prompt = f"""
Fix the logic error in this SQL query:

User Question: {user_query}
Current SQL: {sql_query}
Logic Error: {error}
Database Schema: {schema_desc}

Generate SQL that correctly answers the user's question.
Return only the corrected SQL query.
"""
    
    try:
        result = llm.invoke(correction_prompt)
        corrected = clean_sql_output(result.content)
        return corrected if corrected else sql_query
    except:
        return sql_query


# Utility functions

def extract_table_names(sql_query: str) -> List[str]:
    """Extract table names from SQL query"""
    tables = []
    sql_upper = sql_query.upper()
    
    # Find FROM clause
    from_match = re.search(r'FROM\s+(\w+)', sql_upper)
    if from_match:
        tables.append(from_match.group(1).lower())
    
    # Find JOIN clauses
    join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
    for match in join_matches:
        tables.append(match.lower())
    
    return list(set(tables))


def extract_column_references(sql_query: str) -> List[tuple]:
    """Extract column references with their table aliases"""
    column_refs = []
    
    # Find patterns like alias.column
    pattern = r'(\w+)\.(\w+)'
    matches = re.findall(pattern, sql_query, re.IGNORECASE)
    
    for alias, column in matches:
        column_refs.append((alias.lower(), column.lower()))
    
    return list(set(column_refs))


def find_table_for_alias(sql_query: str, alias: str, table_structure: Dict) -> str:
    """Find actual table name for an alias"""
    sql_upper = sql_query.upper()
    alias_upper = alias.upper()
    
    # Look for pattern: table_name alias
    pattern = rf'(\w+)\s+{alias_upper}\b'
    match = re.search(pattern, sql_upper)
    
    if match:
        table_name = match.group(1).lower()
        if table_name in table_structure:
            return table_name
    
    return None


def create_schema_description(table_structure: Dict) -> str:
    """Create human-readable schema description"""
    desc_parts = []
    
    for table, info in table_structure.items():
        desc_parts.append(f"Table {table}:")
        desc_parts.append(f"  Columns: {', '.join(info['columns'])}")
    
    return "\n".join(desc_parts)


def clean_sql_output(output: str) -> str:
    """Clean LLM output to extract just the SQL"""
    # Remove markdown code blocks
    if "```sql" in output:
        output = output.split("```sql")[1].split("```")[0].strip()
    elif "```" in output:
        output = output.split("```")[1].split("```")[0].strip()
    
    # Remove common prefixes
    prefixes = ["SQL:", "Query:", "Corrected SQL:", "Fixed SQL:"]
    for prefix in prefixes:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
    
    return output.strip()