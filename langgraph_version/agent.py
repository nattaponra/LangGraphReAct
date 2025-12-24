import os
import sys
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import GOOGLE_GEMINI_API_KEY, GOOGLE_GEMINI_MODEL_NAME
from langgraph_version.tools import all_tools


# Define the state schema
class AgentState(TypedDict):
    """State schema for the ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

You have access to the following tools:
1. search_knowledge_base: Search internal company knowledge base for policies, documents, and internal information
2. web_search: Search the internet for current events, external information, or real-time data

Guidelines:
- First, try to understand what type of information the user needs
- Use search_knowledge_base for internal/company-related queries
- Use web_search for external/current information
- You can use multiple tools if needed to gather comprehensive information
- Always provide clear, helpful answers based on the information gathered
"""


def create_react_agent():
    """Create and return a compiled ReAct agent graph."""

    # Initialize the LLM with tools
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_GEMINI_MODEL_NAME,
        google_api_key=GOOGLE_GEMINI_API_KEY,
        temperature=0.2,
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # Define the assistant node
    def assistant(state: AgentState) -> dict:
        """The assistant node that calls the LLM."""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(all_tools))

    # Add edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,  # Routes to "tools" if tool call, else to END
    )
    builder.add_edge("tools", "assistant")  # Loop back after tool execution

    # Compile the graph
    graph = builder.compile()

    return graph


class ReActAgent:
    """Wrapper class for the LangGraph ReAct agent."""

    def __init__(self, enable_logging: bool = True):
        self.graph = create_react_agent()
        self.enable_logging = enable_logging

    def run(self, user_input: str) -> str:
        """
        Run the agent with the given user input.

        Args:
            user_input: The user's question or request

        Returns:
            The agent's final response as a string
        """
        if self.enable_logging:
            print(f"\n{'='*50}")
            print(f"User Query: {user_input}")
            print(f"{'='*50}\n")

        # Create initial state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Extract the final response
        messages = final_state["messages"]

        if self.enable_logging:
            print("\n--- Message History ---")
            for msg in messages:
                msg_type = type(msg).__name__
                content = getattr(msg, 'content', str(msg))
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"[{msg_type}] Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
                else:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"[{msg_type}] {preview}")
            print("-" * 30)

        # Return the last AI message content
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                # Skip messages that have tool calls (intermediate steps)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    continue

                content = msg.content
                # Handle case where content is a list of dicts (Gemini format)
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and 'text' in part:
                            text_parts.append(part['text'])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return '\n'.join(text_parts) if text_parts else str(content)
                return content

        return "Unable to generate a response."

    def stream(self, user_input: str):
        """
        Stream the agent execution for real-time updates.

        Args:
            user_input: The user's question or request

        Yields:
            State updates as the agent processes
        """
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        for state in self.graph.stream(initial_state):
            yield state
