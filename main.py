from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any,Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 
from nodes.entity_clarity_node import entity_resolver_node,load_table_columns_pg,build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node
import psycopg2

load_dotenv()


conn = psycopg2.connect(
    host="localhost",
    dbname="haldiram",
    user="postgres",
    password="12345678"
)

# ---------- Graph State ----------
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

# Graph
graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("question_validator", question_validator)
graph.add_node("entity_resolver", entity_resolver_node)
graph.add_node("sql_generator", sql_agent_node)
graph.add_node("validator_sql", validator_agent)
graph.add_node('executor_sql',sql_executor_node)
graph.add_node("summarized_results", summarized_results_node)


graph.set_entry_point("question_validator")


def route_question(state):
    if state.get("route_decision") == "entity_resolver":
        return "entity_resolver"
    else:
        return "summarized_results"  # Invalid queries go directly to summary

graph.add_conditional_edges(
    "question_validator",
    route_question,
    {
        "entity_resolver": "entity_resolver",
        "summarized_results": "summarized_results"
    }
)


graph.add_edge("entity_resolver", "sql_generator")
graph.add_edge("sql_generator", "validator_sql")
graph.add_edge("validator_sql", "executor_sql") 
graph.add_edge("executor_sql","summarized_results") 

# End after summarization
graph.add_edge("summarized_results", END)

compiled = graph.compile()

png_data = compiled.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])

# Build the catalog with actual values
catalog = build_catalog(conn, table_columns)
result = compiled.invoke({
    "user_query": "total sales of bhujia by vh trading",
    "catalog": catalog,              # supply schema/catalog
    "table_columns": table_columns   # discovered table → columns mapping
})

print("\n--- FINAL RESULT FOR USER ---")
# print(result["validated_sql"])
# print(result['validation_status'])
# print(result['execution_result'])
print(result['final_output'])
