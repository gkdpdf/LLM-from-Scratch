import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Any, Dict,TypedDict,Optional,List

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

def sql_executor_node(state: GraphState):
    validated_sql = state.get("validated_sql", "")
    
    print(f"🔍 DEBUG - Executing SQL: {validated_sql}")
    
    if not validated_sql:
        return {
            "execution_result": None,
            "execution_status": "failed",
            "execution_error": "No validated SQL query to execute"
        }
    
    try:
        connection_params = {
            'host': 'localhost',
            'database': 'haldiram',
            'user': 'postgres',
            'password': '12345678',
            'port': 5432
        }
        
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print(f"📋 Executing: {validated_sql}")
        cursor.execute(validated_sql)
        
        if cursor.description:
            results = cursor.fetchall()
            result_data = [dict(row) for row in results]
            print(f"✅ Success! Found {len(result_data)} rows")
            if result_data:
                print(f"📊 Sample: {result_data[0]}")
        else:
            result_data = {"affected_rows": cursor.rowcount}
            print(f"✅ Success! Affected rows: {cursor.rowcount}")
        
        cursor.close()
        conn.close()
        
        return {
            "execution_result": result_data,
            "execution_status": "success",
            "execution_error": None
        }
        
    except psycopg2.Error as e:
        error_details = f"PostgreSQL Error: {e.pgcode} - {e.pgerror}" if hasattr(e, 'pgcode') else str(e)
        print(f"❌ Database error: {error_details}")
        return {
            "execution_result": None,
            "execution_status": "failed",
            "execution_error": error_details
        }
    except Exception as e:
        print(f"❌ General error: {str(e)}")
        return {
            "execution_result": None,
            "execution_status": "failed",
            "execution_error": f"Execution error: {str(e)}"
        }