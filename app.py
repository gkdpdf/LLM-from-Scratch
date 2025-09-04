import gradio as gr
import psycopg2
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os 
import threading
import queue
import time
from nodes.entity_clarity_node import entity_resolver_node, load_table_columns_pg, build_catalog
from nodes.sql_creator_node import sql_agent_node
from nodes.validator_sql import validator_agent
from nodes.question_clean import question_validator
from nodes.summarized_result import summarized_results_node
from nodes.executor_sql import sql_executor_node

load_dotenv()

# Global variables for human-in-the-loop
user_response_queue = queue.Queue()
waiting_for_input = False
current_options = []
current_question = ""

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="haldiram",
        user="postgres",
        password="12345678"
    )

# Graph State
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

# Modified input function that works with Gradio
def gradio_input(prompt):
    global waiting_for_input, current_question, current_options
    
    print(f"Human input needed: {prompt}")
    
    # Extract options from prompt
    lines = prompt.split('\n')
    options = []
    for line in lines:
        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
            options.append(line.strip())
    
    current_question = prompt
    current_options = options
    waiting_for_input = True
    
    # Wait for user response
    while waiting_for_input:
        try:
            response = user_response_queue.get(timeout=1)
            waiting_for_input = False
            return response
        except queue.Empty:
            continue
    
    return "1"  # Default response

# Override the built-in input function
import builtins
builtins.input = gradio_input

# Build the graph
def create_graph():
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("question_validator", question_validator)
    graph.add_node("entity_resolver", entity_resolver_node)
    graph.add_node("sql_generator", sql_agent_node)
    graph.add_node("validator_sql", validator_agent)
    graph.add_node('executor_sql', sql_executor_node)
    graph.add_node("summarized_results", summarized_results_node)
    
    # Set entry point
    graph.set_entry_point("question_validator")
    
    # Routing function
    def route_question(state):
        if state.get("route_decision") == "entity_resolver":
            return "entity_resolver"
        else:
            return "summarized_results"
    
    # Add edges
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
    graph.add_edge("executor_sql", "summarized_results") 
    graph.add_edge("summarized_results", END)
    
    return graph.compile()

# Initialize components
print("Initializing SQL Query Assistant...")
conn = get_db_connection()
table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
catalog = build_catalog(conn, table_columns)
compiled_graph = create_graph()
conn.close()

print("System initialized successfully!")

# Main query processing function
def process_query(user_query, show_sql=False, show_details=False):
    """Process user query and return results"""
    global waiting_for_input, current_question, current_options
    
    if not user_query.strip():
        return "Please enter a question.", "", ""
    
    try:
        print(f"Processing query: {user_query}")
        
        # Reset human-in-the-loop state
        waiting_for_input = False
        current_question = ""
        current_options = []
        
        # Get fresh database connection
        conn = get_db_connection()
        table_columns = load_table_columns_pg(conn, ["tbl_shipment", "tbl_primary", "tbl_product_master"])
        catalog = build_catalog(conn, table_columns)
        conn.close()
        
        # Process the query in a separate thread
        def run_query():
            return compiled_graph.invoke({
                "user_query": user_query,
                "catalog": catalog,
                "table_columns": table_columns
            })
        
        # Start processing
        query_thread = threading.Thread(target=lambda: setattr(query_thread, 'result', run_query()))
        query_thread.start()
        
        # Wait for completion or human input needed
        while query_thread.is_alive():
            if waiting_for_input:
                # Return current state and show input interface
                return (
                    "I need your help to clarify something...",
                    "",
                    ""
                )
            time.sleep(0.1)
        
        # Get the result
        result = query_thread.result
        
        # Extract results
        final_answer = result.get('final_output', 'No result generated.')
        generated_sql = result.get('validated_sql', 'No SQL generated.')
        
        # Create detailed info
        details = create_detailed_info(result)
        
        # Return based on user preferences
        if show_sql and show_details:
            return final_answer, f"```sql\n{generated_sql}\n```", details
        elif show_sql:
            return final_answer, f"```sql\n{generated_sql}\n```", ""
        elif show_details:
            return final_answer, "", details
        else:
            return final_answer, "", ""
            
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return error_msg, "", ""

def check_for_human_input():
    """Check if human input is needed and return visibility states"""
    global waiting_for_input, current_question, current_options
    
    if waiting_for_input:
        return True, current_question, current_options if current_options else ["Please wait..."]
    else:
        return False, "", ["Please wait..."]

def submit_user_choice(choice):
    """Handle user choice submission"""
    global waiting_for_input, user_response_queue
    
    if waiting_for_input and choice:
        # Extract number from choice
        if choice.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
            choice_num = choice.split('.')[0]
        else:
            choice_num = choice
        
        user_response_queue.put(choice_num)
        waiting_for_input = False
        return "Choice submitted! Processing continues..."
    else:
        return "No input needed right now."

def create_detailed_info(result):
    """Create detailed processing information"""
    details = []
    
    validation_status = result.get('validation_status', 'unknown')
    if validation_status == 'valid':
        details.append("Validation: SQL query validated successfully")
    elif validation_status == 'corrected':
        details.append("Validation: SQL query was corrected automatically")
    else:
        details.append(f"Validation: {validation_status}")
    
    execution_status = result.get('execution_status', 'unknown')
    if execution_status == 'success':
        execution_result = result.get('execution_result', [])
        if isinstance(execution_result, list):
            details.append(f"Execution: Successfully retrieved {len(execution_result)} records")
        else:
            details.append("Execution: Query executed successfully")
    else:
        execution_error = result.get('execution_error', 'Unknown error')
        details.append(f"Execution: {execution_error}")
    
    route = result.get('route_decision', 'unknown')
    details.append(f"Route: {route}")
    
    resolved = result.get('resolved', {})
    if resolved:
        intent = resolved.get('intent', 'Not identified')
        entities = resolved.get('entities', [])
        details.append(f"Intent: {intent}")
        if entities:
            details.append(f"Entities: {', '.join(entities)}")
    
    return "\n".join(details)

# Sample queries for quick testing
sample_queries = [
    "How many total sales in the last month?",
    "Show me all products with Bhujia", 
    "Sales of Delhi in last 3 months",
    "Takatak sales in last two months",
    "How many distributors sold more than 5 distinct products?",
    "What are the top selling products?"
]

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="SQL Query Assistant", theme=gr.themes.Soft()) as iface:
        
        gr.Markdown("""
        # SQL Query Assistant
        ### Ask questions about your data in natural language!
        
        This assistant helps you query your Haldiram database using plain English. 
        Just type your question and get instant answers!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    label="Ask your question",
                    placeholder="e.g., How many sales in Delhi last month?",
                    lines=2
                )
                
                with gr.Row():
                    show_sql = gr.Checkbox(label="Show Generated SQL", value=False)
                    show_details = gr.Checkbox(label="Show Processing Details", value=False)
                
                submit_btn = gr.Button("Get Answer", variant="primary", size="lg")
                
                # Human input section
                gr.Markdown("---")
                gr.Markdown("### Assistant Needs Help")
                
                human_question = gr.Textbox(
                    label="Question from Assistant",
                    interactive=False,
                    lines=6,
                    visible=False,
                    placeholder="The assistant will ask for clarification here when needed..."
                )
                
                user_choice = gr.Radio(
                    label="Please select your choice:",
                    choices=["Waiting for question..."],
                    interactive=True,
                    visible=False
                )
                
                with gr.Row():
                    choice_submit = gr.Button(
                        "Submit My Choice", 
                        variant="secondary", 
                        visible=False,
                        size="lg"
                    )
                
                choice_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    visible=False
                )
                
                # Sample queries
                with gr.Accordion("Sample Questions", open=False):
                    gr.Markdown("Click on any sample question to try it:")
                    for query in sample_queries:
                        btn = gr.Button(f"{query}", size="sm")
                        btn.click(fn=lambda q=query: q, outputs=[user_input])
            
            with gr.Column(scale=3):
                gr.Markdown("### Results")
                
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=6,
                    max_lines=15,
                    interactive=False
                )
                
                sql_output = gr.Code(
                    label="Generated SQL",
                    language="sql",
                    visible=False,
                    interactive=False
                )
                
                details_output = gr.Textbox(
                    label="Processing Details",
                    lines=8,
                    visible=False,
                    interactive=False
                )
        
        # Event handlers
        def update_visibility(show_sql_val, show_details_val):
            return (
                gr.update(visible=show_sql_val),
                gr.update(visible=show_details_val)
            )
        
        show_sql.change(
            fn=update_visibility,
            inputs=[show_sql, show_details],
            outputs=[sql_output, details_output]
        )
        
        show_details.change(
            fn=update_visibility,
            inputs=[show_sql, show_details],
            outputs=[sql_output, details_output]
        )
        
        # Submit and process query
        submit_btn.click(
            fn=process_query,
            inputs=[user_input, show_sql, show_details],
            outputs=[answer_output, sql_output, details_output]
        )
        
        # Periodic check for human input needed
        def update_human_input_display():
            visible, question, choices = check_for_human_input()
            if visible:
                return (
                    gr.update(visible=True, value=question),
                    gr.update(visible=True, choices=choices, value=choices[0] if choices else ""),
                    gr.update(visible=True),
                    gr.update(visible=True, value="Waiting for your selection...")
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        # Timer to check for human input every second
        timer = gr.Timer(value=1.0)
        timer.tick(
            fn=update_human_input_display,
            outputs=[human_question, user_choice, choice_submit, choice_status]
        )
        
        # Handle choice submission
        choice_submit.click(
            fn=submit_user_choice,
            inputs=[user_choice],
            outputs=[choice_status]
        )
        
        user_input.submit(
            fn=process_query,
            inputs=[user_input, show_sql, show_details],
            outputs=[answer_output, sql_output, details_output]
        )
        
        gr.Markdown("""
        ---
        **How it works**: 
        1. Ask your question in the text box above
        2. If the assistant needs clarification, it will show options in the "Assistant Needs Help" section
        3. Select your preferred option and click "Submit My Choice"
        4. The assistant will continue processing and show your answer
        """)
    
    return iface

# Run the app
if __name__ == "__main__":
    print("Starting Gradio interface...")
    
    app = create_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )