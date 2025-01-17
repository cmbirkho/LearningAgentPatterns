
import os
from typing import TypedDict, List, Union

# LangChain & LangGraph imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool, BaseTool
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

import sqlite3

from flask import Flask, request, jsonify
app = Flask(__name__)



def find_best_supplier(suppliers):
    """
    Logic to find the best supplier
    """
    # simplified logic
    best_supplier = min(suppliers, key=lambda x: x['price_per_unit'])
    return f"Supplier: {best_supplier['name']}, Price: {best_supplier['price_per_unit']}, Delivery Time: {best_supplier['delivery_time']} days"


class SQLQueryTool(BaseTool):
    name: str = "SQLQueryTool"
    description: str = "Executes SQL queries on the company database."

    def _run(self, query: str) -> str:
        conn = sqlite3.connect("example.db")
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            if results:
                return str(results)
            else:
                return "No results found."
        except Exception as e:
            conn.close()
            return f"Error: {str(e)}"


class ProcurementState(TypedDict):
    """
    Tracks the conversation and the agent's current state.
    """
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]


class ProcurementAgent:

    def __init__(self):
        """
        Initialize the procurement agent:
          1. Retrieve the OpenAI API key from the environment (safer than hardcoding).
          2. Create the LLM and set up the specialized LangChain agent with tools.
          3. Create a LangGraph workflow and compile it into a stateful agent.
        """
        # Load from environment or fallback to a placeholder key
        os.environ["OPENAI_API_KEY"]

        # This LLM will be used by both the LangGraph workflow and the LangChain agent
        self.llm = ChatOpenAI(model="gpt-4")

        # 1) Set up the specialized tools
        self.tools = self.setup_tools()

        # 2) Initialize a LangChain agent that can use those tools
        #    Here we use the Chat Conversational React agent as an example
        memory = MemorySaver()
        self.langchain_agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            # checkpointer=memory
        )

        # 3) Create the LangGraph workflow and compile it
        self.graph = self.create_graph().compile()
        


    def setup_tools(self):
        """
        Return a list of specialized tools the LangChain agent can use.
        """
        tools = [
            Tool(
                name="Find_Best_Supplier_Tool",
                func=find_best_supplier,
                description="Used to find the best supplier based on price and delivery time.",
            ),
            Tool(
                name="SQL_Query_Tool",
                func=SQLQueryTool(),
                description=(
                    "A tool for executing SQL queries on the company database "
                    "to access the data needed for Find_Best_Supplier_Tool. "
                    "There is a table called suppliers that you can SELECT * FROM suppliers "
                    "to retrieve the required data. The structure of suppliers table is: "
                    "name, item, price_per_unit, delivery_time."
                ),
            )
        ]
        return tools

    def create_graph(self) -> StateGraph:
        """
        Define a simple workflow with two steps:
        1. receive_request
        2. process_request
        """
        workflow = StateGraph(ProcurementState)

        workflow.add_node("receive_request", self.receive_request)
        workflow.add_node("process_request", self.process_request)
        workflow.add_node("tools", ToolNode(tools=self.setup_tools()))

        # Define the edges (transitions) between nodes
        workflow.add_edge("receive_request", "process_request")
        workflow.add_edge("tools", "process_request")

        # Set entry and finish points for the workflow
        workflow.set_entry_point("receive_request")
        workflow.set_finish_point("process_request")

        return workflow

    def receive_request(self, state: ProcurementState) -> ProcurementState:
        """
        Takes the incoming user request (currently just a placeholder),
        appends it to the conversation, and transitions to the next state.
        """
        # For now, we’ll hardcode an incoming message. In a real app, you'd capture user input.
        request_data = "Hello"
        new_message = HumanMessage(content=request_data)

        return {"messages": state["messages"] + [new_message]}

    def process_request(self, state: ProcurementState) -> ProcurementState:
        """
        Uses the LangChain agent (with tools) to process the user's request.
        """
        # Use the specialized agent to parse the conversation and decide if it needs a tool
        response_text = self.langchain_agent.invoke({"messages": state["messages"]})

        return response_text

    def workflow_run(self):
        """
        Runs the workflow defined by the graph.
        """
        # Start with a system message to set the context for the model
        initial_state = ProcurementState(
            messages=[SystemMessage(content="You are a procurement assistant.")]
        )

        # Execute the workflow defined in our graph
        for state in self.graph.invoke(initial_state):
            print(state)


if __name__ == "__main__":
    agent = ProcurementAgent()
    agent.chat()
