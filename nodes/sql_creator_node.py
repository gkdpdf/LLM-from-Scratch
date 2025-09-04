from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any,Optional

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:12345678@localhost:5432/haldiram")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class SQL expert. 
Convert natural language into **valid SQL queries** only.

### Available Schema:
{annotated_schema}

### Database Relationships:
{relationships}

Rules:
- Use only listed tables/columns.
- Apply relationships correctly when joining.
- Return the final SQL in a ```sql block.
"""),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

sql_agent = create_openai_functions_agent(
    llm=llm,
    tools=toolkit.get_tools(),
    prompt=sql_prompt
)

sql_agent_executor = AgentExecutor(agent=sql_agent, tools=toolkit.get_tools(), verbose=True)

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

def sql_agent_node(state: GraphState):
    resolved = state["resolved"]
    annotated_schema = state["annotated_schema"]

    query_input = f"""
Generate PostGreSQL for the following:
Don't create columns and tables on your own

User Query: {state['user_query']}
Intent: {resolved.get('intent')}
Entities: {resolved.get('entities')}
Filters: {resolved.get('filters')}
Target Table: {resolved.get('table')}
Columns: {resolved.get('columns')}

Schema:
{annotated_schema}
"""

    # FIX: pass everything expected by the template
    result = sql_agent_executor.invoke({
        "input": query_input,
        "annotated_schema": annotated_schema,
        "relationships": ""   # <-- if you don't have relationships, pass empty
    })

    # Extract just the SQL query from the result
    sql_query = result.get("output", "")  # or result["output"] if you're sure it exists
    
    # Clean up the SQL query if needed (remove extra text, formatting, etc.)
    # You might need to parse this depending on what your agent returns
    if "```sql" in sql_query:
        # Extract SQL from markdown code blocks
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        # Extract from generic code blocks
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    return {"sql_result": sql_query}