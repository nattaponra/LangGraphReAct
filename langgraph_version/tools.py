from langchain_core.tools import tool
from duckduckgo_search import DDGS
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import rag_search_context


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base (RAG system) for relevant information.
    Use this tool when you need to find information from company documents,
    policies, or internal data.

    Args:
        query: The search query or keywords to find relevant documents

    Returns:
        A string containing the relevant documents found
    """
    docs = rag_search_context(query, top_k=2)

    if docs:
        results = []
        for doc in docs:
            results.append(f"Title: {doc['title']}\nContent: {doc['content']}")
        return "\n\n---\n\n".join(results)
    else:
        return "No relevant information found in the internal knowledge base."


@tool
def web_search(query: str) -> str:
    """
    Search the internet using DuckDuckGo for current/external information.
    Use this tool when you need real-time information, news, or data
    that might not be in the internal knowledge base.

    Args:
        query: The search query to find information on the web

    Returns:
        A string containing the search results from the web
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No relevant information found on the web."

        formatted_results = []
        for r in results:
            title = r.get('title', 'No title')
            body = r.get('body', 'No content')
            href = r.get('href', '')
            formatted_results.append(f"Title: {title}\nContent: {body}\nURL: {href}")

        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {e}"


# List of all tools for the agent
all_tools = [search_knowledge_base, web_search]
