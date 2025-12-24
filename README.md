# LangGraph ReAct Agent

A ReAct (Reasoning and Acting) Agent implementation in Python, featuring both a custom implementation and a LangGraph-based version.

## Overview

This project demonstrates two approaches to building a ReAct Agent:

1. **Original Version** - Custom Python class-based implementation with manual loop control
2. **LangGraph Version** - Graph-based implementation using LangChain's LangGraph framework

Both versions use Google Gemini as the LLM and include:
- Internal knowledge base search (RAG)
- Web search via DuckDuckGo

## Project Structure

```
LangGraphReAct/
├── main.py                 # Entry point for original version
├── react_agent.py          # Original ReAct agent implementation
├── agent_actions.py        # Action functions for original version
├── rag.py                  # RAG system (mock knowledge base)
├── constant.py             # Configuration constants
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── data/
│   └── mock_rag_document.md
└── langgraph_version/      # LangGraph implementation
    ├── __init__.py
    ├── main.py
    ├── agent.py
    └── tools.py
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd LangGraphReAct
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file with your API key:

```
GOOGLE_GEMINI_API_KEY=your_api_key_here
GOOGLE_GEMINI_MODEL_NAME=gemini-2.5-flash
```

## Usage

### Original Version

```bash
python main.py
```

### LangGraph Version

```bash
python langgraph_version/main.py
```

### Programmatic Usage

```python
# Original version
from react_agent import ReActAgent

agent = ReActAgent()
answer = agent.run("What are the employee benefits?")
print(answer)

# LangGraph version
from langgraph_version import ReActAgent

agent = ReActAgent()
answer = agent.run("What are the employee benefits?")
print(answer)
```

## Architecture Comparison

### Original Version

- Manual `for` loop with configurable max steps
- JSON-based action selection via LLM prompting
- Explicit if/elif routing for tool execution
- Observations stored in a list

### LangGraph Version

- StateGraph with nodes and edges
- Native tool calling via `bind_tools()`
- Automatic routing with `tools_condition`
- Message-based state management

## Available Tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Search internal RAG knowledge base |
| `web_search` | Search the internet via DuckDuckGo |

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_GEMINI_API_KEY` | Google Gemini API key |
| `GOOGLE_GEMINI_MODEL_NAME` | Model name (default: gemini-2.5-flash) |

### Agent Parameters

```python
# Original version
agent = ReActAgent(
    max_steps=5,        # Maximum reasoning steps
    enable_logging=True # Enable debug logging
)

# LangGraph version
agent = ReActAgent(
    enable_logging=True # Enable debug logging
)
```

## Debugging

VS Code launch configurations are provided:

- **Python: Main (Original)** - Debug the original implementation
- **Python: LangGraph Version** - Debug the LangGraph implementation
- **Python: Current File** - Debug any open Python file

## Dependencies

- google-generativeai - Google Gemini SDK
- duckduckgo-search - Web search
- python-dotenv - Environment variable management
- langgraph - LangGraph framework
- langchain-core - LangChain core components
- langchain-google-genai - LangChain Google Gemini integration

## License

MIT
