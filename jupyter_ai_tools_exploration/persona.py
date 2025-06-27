from typing import Any
import uuid

from jupyterlab_chat.models import Message

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from jupyter_ai.history import YChatHistory
from jupyter_ai.personas import BasePersona, PersonaDefaults
from jupyter_ai.personas.jupyternaut.prompt_template import (
    JUPYTERNAUT_PROMPT_TEMPLATE,
    JupyternautVariables,
)

from .agent import create_agent


class TestPersona(BasePersona):
    """The Test persona, using natural language to do various tasks in a notebook."""

    agent: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = create_agent(
            get_workspace_path=self.get_workspace_dir, log=self.log
        )
        self.thread_id = str(uuid.uuid4())

    @property
    def defaults(self):
        return PersonaDefaults(
            name="TestPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="The test agent that performs different actions in a notebook.",
            system_prompt="...",
        )

    async def process_message(self, message: Message):
        at_mention = f"@{self.name}"
        msg = message.body.strip().replace(at_mention, "")
        
        if msg:
            await self.run_notebook_agent(msg)
        else:
            self.send_message(
                "Empty prompt received. Please try again with a valid prompt."
            )

    # This is not used in the TestPersona
    def build_runnable(self) -> Any:
        llm = self.config_manager.lm_provider(**self.config_manager.lm_provider_params)
        runnable = JUPYTERNAUT_PROMPT_TEMPLATE | llm | StrOutputParser()
        runnable = RunnableWithMessageHistory(
            runnable=runnable,  #  type:ignore[arg-type]
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=0),
            input_messages_key="input",
            history_messages_key="history",
        )
        return runnable

    async def run_notebook_agent(self, message: Message):
        self.send_message(f"The {self.name} is processing your request...\n")
        config = {"configurable": {"thread_id": self.thread_id}}

        stream = extract_content_stream(
            self.agent.astream({"messages": [message]}, config, stream_mode="messages"),
            log=self.log,
        )
        await self.stream_message(stream)


async def extract_content_stream(original_stream, log):
    """Handles messages from the stream when stream_mode is 'messages'"""
    async for message, _ in original_stream:
        log.debug(message)
        if isinstance(message.content, str):
            yield message.content
        elif isinstance(message.content, list):
            for content in message.content:
                if "type" in content and content["type"] == "text":
                    yield content["text"]


async def extract_content_stream_with_values(original_stream, log):
    """Handles messages from the stream when stream_mode is 'values'"""
    async for s in original_stream:
        message = s["messages"][-1]
        log.debug(s)
        if isinstance(message, tuple):
            yield str(message)
        else:
            yield message.pretty_repr()
