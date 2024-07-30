"""
This Python module defines classes and methods for generating and handling queries using Azure AI Studio. 
It provides an abstract base class, `PromptGenerator`, which sets up the foundation for query generation and 
response handling, and a concrete implementation, `FunctionCallingGenerator`, which is specifically tailored 
for generating database queries and utilizing functions and tools provided by the AI model. 

The module is designed to be utilized in applications that interact with Azure's AI services to perform tasks 
like database querying, processing natural language inputs, and other AI-driven operations. 

The module makes use of the `httpx` library for asynchronous HTTP requests and employs Azure Log Handler for logging. 
It also demonstrates best practices in async programming, error handling, and interaction with external AI services.
"""

from __future__ import annotations

import json
import logging
from string import Template
from typing import Dict, List, Optional, Union

from aistudio_requests.__base import BaseGenerator
from aistudio_requests.prompts import DEFAULT_SYSTEM_MESSAGE
from aistudio_requests.schemas import AzureAIFunction, AzureAIMessage, AzureAIRequest, AzureAITool

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class PromptGenerator(BaseGenerator):
    """
    Abstract base class representing a query generator using Azure AI Studio.

    This class provides the foundational methods and properties for generating queries
    and handling responses from the Azure AI Studio. It applies the "Template Method" pattern
    on the prepare_request method.

    Methods:
        send_request(prompt_template: PromptTemplate, parameters: Dict[str, Union[str, float, int]], complete_response: bool=False)
            Asynchronously sends a request to the Azure AI Studio using the provided parameters.
    """

    async def send_request(
        self,
        prompt: str,
        parameters: Dict[str, Union[str, float, int]],
        *,
        complete_response: bool = False
    ):
        """
        Asynchronously send a request to the Azure AI Studio using the provided parameters.
        Constructs and sends the prompt request and processes the response.

        Args:
            prompt_template (PromptTemplate): The prompt template to generate the prompt.
            parameters (Dict[str, Union[str, float, int]]): Additional parameters for the request.
            complete_response (bool, optional): Whether to return the complete response. Defaults to False.

        Returns:
            str: The result of the prompt from the Azure AI Studio, or None if the request was unsuccessful.
        """

        messages: List[AzureAIMessage] = [
            AzureAIMessage(
                role="system",
                content=[{"type": "text", "text": self.system_message}],
            ),
            AzureAIMessage(
                role="user",
                content=[{"type": "text", "text": prompt}],
            ),
        ]

        logger.debug("Sending query to Azure AI Service. Messages: %s \n", messages)
        data = AzureAIRequest(messages=messages, **parameters)  # type: ignore
        json_data = data.model_dump(exclude_unset=True, exclude_none=True)
        logger.debug("Sending data to Azure AI Studio. Data: %s \n", json_data)

        response = await self._request_url(method="post", url=self.aistudio_url, data=json_data)

        logger.info(
            "Query successfully generated. Resources used: %s",
            str(
                {
                    "model": response.get("model", ""),
                    **response.get("usage", {}),
                }
            ),
        )

        if not complete_response:
            response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return response


class FunctionCallingGenerator(BaseGenerator):
    """
    A abstract class to interact with models that have function call, tailored for the usage of functions and tools by the LLM.

    Methods:
        prepare_request(prompt: PromptTemplate) -> str
            Prepares the request for the Azure AI Studio.

        send_request(prompt_template: PromptTemplate, parameters: Dict[str, Union[str, float, int]]) -> PromptTemplate
            Asynchronously sends a request to the Azure AI Studio using the provided parameters.
    """

    def __init__(self, aistudio_url: str, aistudio_key: str, az_monitor: str | None = None) -> None:
        super().__init__(aistudio_url, aistudio_key, az_monitor)
        self._functions: Optional[List[AzureAIFunction]] = None

    @property
    def functions(self) -> List[AzureAIFunction]:
        """
        Get the system message.

        Returns:
            str: The current system message.
        """
        if self._functions is None:
            raise AttributeError("Function Calling should be provided for the request of the .")
        return self._functions

    @functions.setter
    def functions(self, function_calls: List[AzureAIFunction]) -> None:
        """
        Set a new system message.

        Args:
            message (str): The new system message.
        """
        self._functions = function_calls

    @functions.deleter
    def functions(self) -> None:
        """
        Reset the system message to the default value.
        """
        self._functions = None

    async def send_request(
        self,
        prompt: str,
        parameters: Dict[str, str | float | int],
        *,
        complete_response: bool = False,
    ) -> dict:
        """
        Asynchronously send a request to the Azure AI Studio using the provided parameters.
        Constructs and sends the prompt request and processes the response.

        Args:
            prompt_template (PromptTemplate): The prompt template to generate the prompt.
            parameters (Dict[str, Union[str, float, int]]): Additional parameters for the request.

        Returns:
            PromptTemplate: The result of the prompt from the Azure AI Studio as a PromptTemplate object.

        Raises:
            ValueError: If the prompt request could not be generated.
        """

        if self.functions is None:
            raise AttributeError(
                "Function Calling should be provided for the request of the FunctionCallingGenerator."
            )

        self.system_message = Template(DEFAULT_SYSTEM_MESSAGE).safe_substitute(
            functions="PromptTemplate"
        )

        messages: List[AzureAIMessage] = [
            AzureAIMessage(
                role="system",
                content=[{"type": "text", "text": self.system_message}],
            ),
            AzureAIMessage(
                role="user",
                content=[{"type": "text", "text": prompt}],
            ),
        ]

        tools = [
            AzureAITool(type="function", function=_func) for _func in self.functions
        ]

        logger.debug("Sending query to Azure AI Studio. Messages: %s \n", messages)
        data = AzureAIRequest(messages=messages, tools=tools, **parameters)  # type: ignore
        json_data = data.model_dump(exclude_unset=True, exclude_none=True)
        logger.debug("Sending data to Azure AI Studio. Data: %s \n", json_data)

        response = await self._request_url(
            method="post",
            url=self.aistudio_url,
            data=json_data
        )

        logger.info(
            "Query successfully generated. Resources used: %s",
            str(
                {
                    "model": response.get("model", ""),
                    **response.get("usage", {}),
                }
            ),
        )

        if complete_response:
            return response

        tool_calls = response.get("choices", [])[0].get("message", {}).get("tool_calls", [])
        arguments = json.loads(tool_calls[0].get("function", {}).get("arguments", {}))

        return arguments
