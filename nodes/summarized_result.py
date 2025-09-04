
from typing import TypedDict, Dict, Any, List, Optional
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


def summarized_results_node(state: GraphState):
    user_query = state.get("user_query", "")
    execution_result = state.get("execution_result", [])
    execution_status = state.get("execution_status", "")
    
    # Handle invalid queries
    if state.get("validation_status") == "invalid_query":
        return {"final_output": "I can only help with database queries. Please ask about retrieving or analyzing data."}
    
    # Handle execution failures
    if execution_status == "failed":
        return {"final_output": "Sorry, I couldn't retrieve the requested data due to a database error."}
    
    # Generate natural language response from results
    if execution_status == "success" and execution_result:
        # Create prompt for NLP summarization
        summarization_prompt = f"""
Convert this database query result into a natural, conversational answer.

User Question: "{user_query}"
Query Results: {execution_result}

Provide a single, clear sentence that directly answers the user's question using the data.
Be conversational and natural. Don't mention SQL or technical details.

Examples:
- If asked "how many customers are there?" and result is 150 records, answer: "There are 150 customers in the database."
- If asked "who are the top salespeople?" and results show names, answer: "The top salespeople are John, Sarah, and Mike."
- If asked "what's the total revenue?" and result is $50000, answer: "The total revenue is $50,000."

Answer:"""
        
        # Use LLM to generate natural response
        summarization_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=os.getenv('OPENAI_API_KEY'))
        result = summarization_llm.invoke(summarization_prompt)
        
        return {"final_output": result.content.strip()}
    
    # No results case
    elif execution_status == "success" and not execution_result:
        return {"final_output": "No matching records were found for your query."}
    
    # Fallback
    else:
        return {"final_output": "I couldn't process your request. Please try rephrasing your question."}