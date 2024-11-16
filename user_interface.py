from openCHA import openCHA
from agent import orchestrator

# Define available tasks from your agent setup
available_tasks = [
    "daily_preprocessing",
    "retrieval_task",
    "sentiment_analysis",
    "future_mental_health_prediction",
    "generate_recommendations",
]

chat_history = []

def terminal_chat():
    """
    Terminal-based chat interface.
    """
    welcome_message = """
    Welcome to the TrackSense Chatbot! (Your Work Buddy)
    I’m here to help with your mental health queries.

    - Type "exit" to leave the chat.
    - Ask questions like:
      - "How can I manage work stress?"
      - "What steps should I take if I feel overwhelmed?"
      - "What are some tips to improve mental health?"
    """
    print(welcome_message)

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for using the chatbot. Stay safe!")
            break

        # Initialize OpenCHA with your orchestrator
        cha = openCHA(orchestrator=orchestrator)
        response = cha.run(
            user_query,
            chat_history=chat_history,
            available_tasks=available_tasks,
            use_history=True,
        )
        print("Chatbot:", response)

        # Save the conversation to chat history
        chat_history.append((user_query, response))


def web_interface():
    """
    Web-based chat interface.
    """
    cha = openCHA(orchestrator=orchestrator, verbose=True)
    print("Launching the web-based chat interface...")
    cha.run_with_interface()
    # Follow the terminal printout for the URL to access the interface in a browser.


if __name__ == "__main__":
    # Choose between terminal or web interface
    mode = input("Choose interaction mode (terminal/web): ").strip().lower()

    if mode == "terminal":
        terminal_chat()
    elif mode == "web":
        web_interface()
    else:
        print("Invalid choice. Please run the script again and choose 'terminal' or 'web'.")
