from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from .tools import add_cell, delete_cell

DEFAULT_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_PROVIDER = "bedrock_converse"

def create_agent(model: str = DEFAULT_MODEL, model_provider: str = DEFAULT_PROVIDER):
    tools = [add_cell, delete_cell]
    chat_model = init_chat_model(
        model=model, model_provider=model_provider
    )

    agent = create_react_agent(
        model=chat_model,
        tools=tools
    )

    return agent