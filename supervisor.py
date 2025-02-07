

import os
import json
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"]

# Initialize LLM model
model = ChatOpenAI(model_name="gpt-4-turbo")

# Define Supervisor Agent
def supervisor(state: MessagesState) -> Command[Literal["math_agent", "text_agent", END]]:
    """
    Supervisor determines whether to route the task to 'math_agent' or 'text_agent'.
    Uses LLM to classify the task type.
    Supervisor can see the full conversation history as it interacts with text_agent and math_agent
    """

    print(f"Supervisor 1st State: {state}")

    # Invoke the model with the current state messages
    response = model.invoke(state["messages"])
    print(f"Supervisor Response: {response}")
    response_content = json.loads(response.content)
    next_agent = response_content.get("next_agent")
    task = response_content.get("task")

    # Extract next agent decision
    if next_agent not in ["math_agent", "text_agent"]:
        next_agent = END  # If the decision is unclear, end execution

    # Add the model's response to the state.
    # Wrap in a human message, as not all providers allow AI message at the last position of the input messages list 
    response_content = HumanMessage(content=f"Supervisor task: {task}")

    state = {"messages": state["messages"] + [response_content]}
    print(f"Supervisor 2nd State: {state}")
    return Command(
        goto=next_agent,
        update=state
    )

# Define Math Agent
def math_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Math agent processes math-related queries.
    """
    task = state.messages[-1].content
    print(f"Math Agent received task: {task}")  # Debugging print

    response = model.invoke([HumanMessage(content=f"Solve this math problem: {task}")])

    # Add the model's response to the state
    state.add_message(SystemMessage(content=response.content))

    return Command(
        goto="supervisor",
        update=state
    )

# Define Text Agent
def text_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Text agent processes text/language-related queries.
    Text agent can only see the last message from the supervisor.
    """


    text_sys_msg = SystemMessage(
            content=(
                """You are a text agent responsible for solving and answering text related inquiries.

                ### Objective:
                Your goal is to analyze the given task and solve/answer the inquiry. 
                Your response will go to the Supervisor. You can interact with the Supervisor and ask them clarifying questions.
                """
            )
    )
    
    # Get the last message
    last_message = state["messages"][-1]
    print(last_message)
    # Organize the System Message and last message
    state = MessagesState(messages=[text_sys_msg] + [last_message])
    print(f"text_agent 1st state: {state}")

    response = model.invoke(state["messages"])
    response_content = AIMessage(content=response.content)

    state = {"messages": state["messages"] + [response_content]}

    print(f"text_agent 2nd state: {state}")

    return Command(
        goto="supervisor",
        update=state
    )

# Build LangGraph Workflow
def build_graph():
    graph = StateGraph(MessagesState)

    # Define nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("math_agent", math_agent)
    graph.add_node("text_agent", text_agent)

    # Define edges
    graph.add_edge(START, "supervisor")
    graph.add_edge("math_agent", "supervisor")
    graph.add_edge("text_agent", "supervisor")

    return graph.compile()

# Run the LangGraph Workflow
def main():
    workflow = build_graph()

    print("Welcome to the AI Agent Network using LangGraph.")
    print("Type your task or 'exit' to quit.")

    while True:
        user_input = input("Enter your task: ").strip()
        if user_input.lower() == "exit":
            print("Exiting the AI Agent Network. Goodbye!")
            break

        # Start graph execution with user input
        # Add system message for the supervisor who starts the flow
        supervisor_sys_msg = SystemMessage(
            content=(
                """You are a supervisor agent responsible for routing tasks to specialized agents based on the task's nature.

                ### Objective:
                Your goal is to analyze the given task, determine its type, and route it to the appropriate specialized agent or __end__ the conversation if you receive an adequate response from the specialized agent.

                ### Available Specialized Agents:
                - **math_agent**: Handles all math-related inquiries, including calculations, formulas, and problem-solving.
                - **text_agent**: Handles all text-related inquiries, such as text generation, summarization, and analysis.

                ### Response Format:
                You must respond with a valid JSON object in the following structure:
                {
                "next_agent": "<one of: 'math_agent', 'text_agent', '__end__'>",
                "task": "<a concise description of the task for the assigned agent>"
                }
                """
            )
        )
        initial_state = MessagesState(messages=[supervisor_sys_msg] + [HumanMessage(content=user_input)])
        print(initial_state)
        final_state = workflow.invoke(initial_state)

        print("\nFinal Response:")
        print(final_state.messages[-1].content)
        print("-" * 40)