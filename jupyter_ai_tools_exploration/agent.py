from logging import Logger
import os
from typing import Optional
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from jupyter_ai_tools.toolkits.file_system import toolkit as fs_toolkit
from jupyter_ai_tools.toolkits.notebook import toolkit as nb_toolkit

#DEFAULT_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
#DEFAULT_PROVIDER = "bedrock_converse"

DEFAULT_MODEL = "o4-mini-2025-04-16"
DEFAULT_PROVIDER = "openai"

def create_agent(
    model: str = DEFAULT_MODEL,
    model_provider: str = DEFAULT_PROVIDER,
    prompt: Optional[str] = None,
    get_workspace_path: Optional[callable] = None,
    log: Logger = None
):
    
    async def get_project_path(_: bool = True) -> str:
        """
        Returns the absolute path for the current project
        
        Args:
            _: Boolean parameter (unused, required for tool interface)
            
        Returns:
            The absolute path to the current project directory
        """
        if get_workspace_path is not None:
            ws_path = get_workspace_path()
        else:
            ws_path = str(os.getcwd())

        print(f"{ws_path=}")
        return ws_path

    tools = [get_project_path]
    tools += [tool.callable for tool in fs_toolkit.get_tools()]
    tools += [tool.callable for tool in nb_toolkit.get_tools()]

    chat_model = init_chat_model(
        model=model, 
        model_provider=model_provider
    )
    checkpointer = InMemorySaver()
    
    agent = create_react_agent(
        model=chat_model, 
        tools=tools, 
        prompt=prompt, 
        checkpointer=checkpointer
    )

    return agent
