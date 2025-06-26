from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from jupyter_ai_tools.toolkits.file_system import toolkit as fs_toolkit

DEFAULT_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_PROVIDER = "bedrock_converse"

def create_agent(model: str = DEFAULT_MODEL, model_provider: str = DEFAULT_PROVIDER):
    tools = [tool.callable for tool in fs_toolkit.get_tools()]
    chat_model = init_chat_model(
        model=model, model_provider=model_provider
    )

    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model=chat_model,
        tools=tools,
        checkpointer=checkpointer
    )
    
    return agent