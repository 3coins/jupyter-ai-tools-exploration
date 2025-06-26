from typing import Any, Sequence
import uuid

from jupyterlab_chat.models import Message
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    FunctionMessage,
    ChatMessage,
    SystemMessage,
    AIMessageChunk,
)
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
        self.agent = create_agent()
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
        provider_name = self.config_manager.lm_provider.name
        model_id = self.config_manager.lm_provider_params["model_id"]

        variables = JupyternautVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
        )

        at_mention = f"@{self.name}"
        msg = variables.input.strip().replace(at_mention, "")
        if msg:
            await self.run_notebook_agent(msg)
        else:
            self.send_message(
                "Error: Query failed. Please try again with a different query."
            )

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

        async def extract_content_stream(original_stream):
            async for message, _ in original_stream:
                for content in message.content:
                    if "type" in content and content["type"] == "text":
                        yield content['text']

        stream = extract_content_stream(
            self.agent.astream({"messages": [message]}, config, stream_mode="messages")
        )
        await self.stream_message(stream)


def serialize_messages(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    r"""Convert a sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
            Default is "Human".
        ai_prefix: THe prefix to prepend to contents of AIMessages. Default is "AI".

    Returns:
        A single string concatenation of all input messages.

    Raises:
        ValueError: If an unsupported message type is encountered.

    Example:
        .. code-block:: python

            from langchain_core import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ToolMessage):
            role = "Tool"
            m.content = "Completed execution..."
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004

        content = m.content
        # Handle different content types
        if isinstance(content, list):
            # List of strings case
            if all(isinstance(item, str) for item in content):
                content = "".join(content)
            # List of dictionaries with 'type' and 'text' keys case
            elif all(isinstance(item, dict) for item in content):
                result = []
                for item in content:
                    if item.get("type") == "text" and "text" in item:
                        result.append(item["text"])
                    elif (
                        item.get("type") == "tool_use"
                        and "name" in item
                        and "input" in item
                    ):
                        tool_name = item.get("name")
                        tool_input = item.get("input")
                        result.append(
                            f"\n\nCalling **{tool_name}** with inputs: {tool_input}"
                        )
                content = "".join(result)

        message = f"{role}: {content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)
