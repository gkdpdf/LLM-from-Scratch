from typing import TypedDict, List, Dict, Any,Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

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

def question_validator(state: GraphState):
    user_query = state['user_query']
    
    # Create validation prompt to check if query is database-related
    validation_prompt = f"""
Analyze the following user query to determine if it's a valid database query that can be answered with SQL:

User Query: "{user_query}"

Consider:
1. Does it ask for data retrieval, filtering, or analysis?
2. Does it mention entities that could be database tables/columns?
3. Is it asking for counts, sums, averages, or other aggregations?
4. Does it request specific records or filtered results?

Examples of VALID queries:
- "Show me all customers from New York"
- "What's the total sales for last month?"
- "List employees with salary > 50000"
- "How many orders were placed today?"

Examples of INVALID queries:
- "What's the weather today?"
- "Tell me a joke"
- "How do I cook pasta?"
- "What's 2+2?"

Respond with:
- "VALID" if it's a database query
- "INVALID" if it's not a database query
"""
    
    # Use simple LLM to validate
    validation_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=os.getenv('OPENAI_API_KEY'))
    result = validation_llm.invoke(validation_prompt)
    validation_result = result.content.strip()
    
    if validation_result.startswith("VALID"):
        # Route to entity resolver
        return {
            "route_decision": "entity_resolver",
            "validation_status": "valid_query"
        }
    else:
        # Route to summarized results (general response)
        general_response = f"I'm designed to help with database queries. Your question '{user_query}' doesn't seem to be related to data retrieval. Please ask questions about data, records, or analytics that I can help you with using SQL queries."
        
        return {
            "route_decision": "summarized_results", 
            "final_output": general_response,
            "validation_status": "invalid_query"
        }