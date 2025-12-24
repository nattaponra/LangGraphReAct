import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_version.agent import ReActAgent


def main():
    """
    Main entry point for the LangGraph ReAct Agent.
    """
    # Create the agent
    agent = ReActAgent(enable_logging=True)

    # Interactive loop
    print("LangGraph ReAct Agent")
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            user_query = input("Query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_query:
                continue

            # Run the agent
            answer = agent.run(user_query)

            print(f"\n{'='*50}")
            print("Final Answer:")
            print(f"{'='*50}")
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_single_query(query: str) -> str:
    """
    Run a single query and return the answer.
    Useful for testing or programmatic use.
    """
    agent = ReActAgent(enable_logging=False)
    return agent.run(query)


if __name__ == "__main__":
    main()
