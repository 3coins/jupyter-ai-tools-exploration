from typing import Any
from pydantic import Field, BaseModel

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


class UserQueryClassifier(BaseModel):
    needs_notebook_action: bool = Field(
        description="Returns True if the request calls for an action to the notebook, else False"
    )


class TestPersona(BasePersona):
    """The Test persona, using natural language to do various tasks in a notebook."""

    agent: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = create_agent(
            model="us.meta.llama4-scout-17b-instruct-v1:0"
        )

    @property
    def defaults(self):
        return PersonaDefaults(
            name="TestPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="The test agent that performs different actions in a notebook.",
            system_prompt="...",
        )

    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]

        runnable = self.build_runnable()
        variables = JupyternautVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name,
        )

        # Check if the prompt is about finance. If so, pass on to agentic workflow, else use default handling
        prompt = variables.input.split(" ", 1)[1]
        llm = self.config.lm_provider(**self.config.lm_provider_params)
        llm = llm.with_structured_output(
            UserQueryClassifier,
        )
        response = llm.invoke(prompt)  # Gets the full AI message response

        # If the message does not ask for a notebook action, proceed with default handling
        if response.needs_notebook_action:  # type:ignore[union-attr]
            msg = variables.input.split(" ", 1)[1].strip()
            if msg:
                # Call the agno_finance function to process the message
                self.run_notebook_agent(msg)
            else:
                self.send_message(
                    "Error: Query failed. Please try again with a different query."
                )
        else:  # If the message is not finance-related, use the default runnable
            variables_dict = variables.model_dump()
            reply_stream = runnable.astream(variables_dict)
            await self.stream_message(reply_stream)

    def build_runnable(self) -> Any:
        llm = self.config.lm_provider(**self.config.lm_provider_params)
        runnable = JUPYTERNAUT_PROMPT_TEMPLATE | llm | StrOutputParser()
        runnable = RunnableWithMessageHistory(
            runnable=runnable,  #  type:ignore[arg-type]
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=0),
            input_messages_key="input",
            history_messages_key="history",
        )
        return runnable

    def run_notebook_agent(self, message: Message):
        self.send_message("The test agent is processing your request...")
        response = self.agent.invoke({
            "messages": [message]
        })

        if response:
            last_message = response["messages"][-1]
            self.send_message(last_message.content)
        else:
            self.send_message(
                "No response received from the test agent, please try again!"
            )
