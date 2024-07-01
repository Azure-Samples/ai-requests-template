"""
This Python module defines classes and methods for generating and handling queries using Azure AI Studio. 
It provides an abstract base class, `PromptGenerator`, which sets up the foundation for query generation and 
response handling, and a concrete implementation, `QueryGenerator`, which is specifically tailored for generating 
database queries. The module is designed to be utilized in applications that interact with Azure's AI services 
to perform tasks like database querying, processing natural language inputs, and other AI-driven operations.

The module makes use of the `httpx` library for asynchronous HTTP requests and employs Azure Log Handler for logging. 
It also demonstrates best practices in async programming, error handling, and interaction with external AI services.
"""

from __future__ import annotations

import asyncio
import logging
import json
from abc import abstractmethod
from string import Template
from typing import Dict, List, Optional

import httpx

from src.schemas import (
    AzureAIMessage,
    AzureAIRequest,
    AzureAITool,
    AzureAIFunction,
    PromptTemplate
)

from src.__base import BaseGenerator
from src.system_message import DEFAULT_SYSTEM_MESSAGE


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class PromptGenerator(BaseGenerator):
    #ToDo: Change to "Builder" Design Pattern to build the request with tools, messages and other parameters
    #ToDo: Change to use openai's package with AOAI Design Pattern to build the request with tools, messages and other parameters
    """
    _summary_: Abstract base class representing a query generator using Azure AI Studio.
    This class provides the foundational methods and properties for generating queries
    and handling responses from the Azure AI Studio.

    Applies the "Template Method" pattern on the prepare_request method
    """

    def __init__(
        self,
        aoai_url: str,
        aoai_key: str,
        az_monitor: Optional[str] = None,
    ) -> None:
        """
        Initialize the PromptGenerator with Azure AI Studio URL, access key, and optional
        Azure Monitor connection string for logging.

        Args:
            aoai_url (str): The URL endpoint for the Azure AI Studio.
            aoai_key (str): Access key for Azure AI Studio authentication.
            az_monitor (Optional[str]): Connection string for Azure Monitor, used for logging.
        """
        super().__init__(aoai_url, aoai_key, az_monitor)
        self.http_client = httpx.AsyncClient(timeout=20000)

    async def close(self):
        """
        Asynchronously close the HTTP client connection.
        """
        await self.http_client.aclose()

    async def _request_url(self, url, method, data=None):
        """
        Asynchronous private method to make an HTTP request using the specified URL,
        method, and optional data payload.

        Args:
            url (str): The URL endpoint to send the request to.
            method (str): The HTTP method to use for the request.
            data (Optional): The data payload to send with the request, if any.

        Returns:
            dict: The JSON response from the request.

        Raises:
            HTTPStatusError: If the response status code indicates an error.
        """
        request_param = {
            "method": method,
            "url": url,
            "headers": self.headers,
        }
        if data:
            request_param["json"] = data
        try:
            response = await self.http_client.request(**request_param)
            response.raise_for_status()
            logger.info("Request Successful: %s", response.status_code)
            self.waiting_time = 0
            return response.json()
        except httpx.NetworkError as exc:
            logger.error("Network Error: %s. Trying Again.", str(exc))
            await asyncio.sleep(0.5)
            return await self._request_url(url, method, data)
        except httpx.HTTPStatusError as exc:
            if response.status_code == 503:
                logger.error("Server unavailable: %s. Trying Again.", str(exc))
                await asyncio.sleep(60)
            elif response.status_code == 429:
                logger.error("Too many requests: %s. Trying Again.", str(exc))
                await asyncio.sleep(self.waiting_time**1.5)
            else:
                logger.critical("Untreated error: %s.", str(exc))
                raise exc
            logger.info("Service was unavailable. Trying again.")
            self.waiting_time+=1
            return await self._request_url(url, method, data)

    @abstractmethod
    async def prepare_request(
        self,
        prompt_template: PromptTemplate
    ) -> str:

        """
        Abstract method to be implemented by subclasses, preparing the request to Azure AI Studio.
        Constructs the query request based on given parameters.

        Args:
            prompt (str): The prompt text to generate the query.
            query_type (str): The type of query to generate.
            db_params (Dict[str, Union[str, List[str]]]): Parameters relevant to the database query.
            programming_language (Optional[str]): Optional programming language for the query.

        Returns:
            Dict[str, str]: The prepared prompt request as a dictionary.
        """

    async def send_request(
        self,
        prompt_template: PromptTemplate,
        parameters: Dict[str, str | float | int],
        *,
        complete_response: bool = False
    ):
        """
        Asynchronously send a request to the Azure AI Studio using the provided parameters.
        Constructs and sends the prompt request and processes the response.

        Args:
            prompt (str): The prompt text to generate the prompt.
            prompt_type (str): The type of prompt to generate.
            db_params (Dict[str, Union[str, List[str]]]): Parameters relevant to the database query.
            parameters (Dict[str, Union[str, float, int]]): Additional parameters for the request.
            programming_language (Optional[str]): Optional programming language for the prompt.

        Returns:
            str: The result of the prompt from the Azure AI Studio.
        """

        prompt_request: str = await self.prepare_request(prompt_template)
        if not prompt_request:
            return None

        messages: List[AzureAIMessage] = [
            AzureAIMessage(
                role="system",
                content=[{"type": "text", "text": self.system_message}],
            ),
            AzureAIMessage(
                role="user",
                content=[{"type": "text", "text": prompt_request}],
            ),
        ]

        logger.debug("Sending query to Azure AI Service. Messages: %s \n", messages)

        data = AzureAIRequest(messages=messages, **parameters)  # type: ignore

        json_data = data.model_dump(exclude_unset=True, exclude_none=True)

        logger.debug("Sending data to Azure AI Studio. Data: %s \n", json_data)

        response = await self._request_url(
            method="post",
            url=self.aoai_url,
            data=json_data
        )

        logger.info(
            "Query successfully generated. Resources used: %s",
            str({
                "model": response.get("model", ""),
                **response.get("usage", {}),
            }),
        )

        if not complete_response:
            response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return response


class FunctionCallingGenerator(PromptGenerator):
    """
    A concrete implementation of the PromptGenerator, tailored for the usage of functions and tools
    by the LLM.
    """

    async def prepare_request(
        self,
        prompt: PromptTemplate
    ):
        """

        """
        query_request = """
            Based on the theme $prompt, generate a question, a groundtruth answer, and a answer that has '50%' chance to be correct.
        """
        return Template(query_request).safe_substitute(prompt=prompt.prompt, function_name=prompt.function_name)

    async def send_request(
        self,
        prompt_template: PromptTemplate,
        parameters: Dict[str, str | float | int]
    ) -> PromptTemplate:
        """
        Asynchronously send a request to the Azure AI Studio using the provided parameters.
        Constructs and sends the prompt request and processes the response.

        Args:
            prompt (str): The prompt text to generate the prompt.
            prompt_type (str): The type of prompt to generate.
            db_params (Dict[str, Union[str, List[str]]]): Parameters relevant to the database query.
            parameters (Dict[str, Union[str, float, int]]): Additional parameters for the request.
            programming_language (Optional[str]): Optional programming language for the prompt.

        Returns:
            str: The result of the prompt from the Azure AI Studio.
        """

        prompt_request: str = await self.prepare_request(prompt_template)
        if not prompt_request:
            raise ValueError("Prompt request could not be generated.")
        
        self.system_message = Template(DEFAULT_SYSTEM_MESSAGE).safe_substitute(functions="PromptTemplate")

        messages: List[AzureAIMessage] = [
            AzureAIMessage(
                role="system",
                content=[{"type": "text", "text": self.system_message}],
            ),
            AzureAIMessage(
                role="user",
                content=[{"type": "text", "text": prompt_request}],
            ),
        ]

        tools = [AzureAITool(
            type='function',
            function=AzureAIFunction(
                name="PromptTemplate",
                description="Format the prompt in a proper QnA JSON format.",
                parameters=PromptTemplate.model_json_schema()
            )
        ),]

        logger.debug("Sending query to Azure AI Studio. Messages: %s \n", messages)
        data = AzureAIRequest(messages=messages, tools=tools, **parameters)  # type: ignore
        json_data = data.model_dump(exclude_unset=True, exclude_none=True)
        logger.debug("Sending data to Azure AI Studio. Data: %s \n", json)

        response = await self._request_url(
            method="post",
            url=self.aoai_url,
            data=json_data
        )

        logger.info(
            "Query successfully generated. Resources used: %s",
            str({
                "model": response.get("model", ""),
                **response.get("usage", {}),
            }),
        )

        tool_calls = response.get("choices", [])[0].get("message", {}).get("tool_calls", [])
        arguments = json.loads(tool_calls[0].get("function", {}).get("arguments", {}))
        
        return PromptTemplate(**arguments)
