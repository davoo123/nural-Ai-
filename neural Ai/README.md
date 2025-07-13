# 🤖 Advanced AI Assistant with Autonomous Learning

An intelligent AI assistant with self-awareness, autonomous learning, reinforcement learning, and voice capabilities.

## Features

- **Self-learning**: Learns from web searches and user interactions
- **Knowledge Base**: Stores information locally for future reference
- **Neural Network**: Uses sentence transformers for semantic similarity
- **Human-like Interaction**: Simulates personality traits for natural conversations
- **Web Search**: Can search the web for information when needed

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Set up environment variables for APIs in a `.env` file

## Usage

Run the AI assistant:
```
python ai_assistant.py
```

## How to Chat with Your AI

- **Type naturally:** You can ask questions or give instructions just as you would with a human assistant. For example:
  - "What's the weather today?"
  - "Explain quantum computing in simple terms."
  - "Remind me to drink water every hour."
- **Follow up:** You can ask follow-up questions or refer to previous topics. The AI will try to remember and relate to your previous interactions.
- **Feedback:** If the AI gives an incorrect or incomplete answer, you can correct it or provide feedback. The AI will learn from this over time.
- **Personality:** The AI is designed to be friendly, helpful, and conversational. You can say hello, ask for advice, or even chat casually.
- **Exit:** Type 'exit', 'quit', or 'bye' to end the conversation at any time.

Interact with the AI by typing your questions. The AI will:
1. First try to answer from its knowledge base
2. If it doesn't know, it will search the web
3. Learn from the interaction for future reference

## Customization

You can customize the AI's personality and learning behavior by modifying the `AILearner` class in `ai_assistant.py`.

## Note

For production use, you should:
1. Add proper API keys for web search services
2. Implement rate limiting and error handling
3. Add user authentication if needed
4. Consider privacy and data storage regulations
