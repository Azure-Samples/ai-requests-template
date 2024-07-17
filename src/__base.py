"""
This module defines the BaseGenerator abstract class and its concrete implementation for generating and handling
requests to the Azure AI Studio. The BaseGenerator class provides a framework for setting up a generator with 
customized request handling logic, including streaming and regular request capabilities. It is designed to be 
extended by subclasses that implement specific behaviors for retrieving context, managing conversation history,
preparing requests, and sending them to a specified model endpoint.

Key functionalities include:
- Setup and initialization with API keys and service URLs.
- Abstract methods enforcing implementation of context retrieval, history management, and request preparation.
- Detailed error handling and retry logic for robustness against network issues and service unavailability.
- Usage of async programming to enhance performance in IO-bound operations.

Usage:
This script is meant to be used as a part of an application that interacts with Azure AI Studio. It requires
external configurations like API keys and service URLs to be provided during initialization.

Dependencies:
- httpx: For asynchronous HTTP requests.
- asyncio: For asynchronous control flow and IO operations.
- opencensus.ext.azure: For logging and monitoring with Azure.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict

import logging
import httpx
import asyncio
from string import Template

from opencensus.ext.azure.log_exporter import AzureLogHandler

from src.prompts import DEFAULT_SYSTEM_MESSAGE, DEFAULT_PROMPT
from src.schemas import PromptTemplate


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BaseGenerator(ABC):
    """
    An abstract base class for generating prompts and sending requests to the Azure AI Studio.

    This class provides a template method and common functionality for subclasses that need to interact
    with the Azure AI Studio service. Subclasses must implement abstract methods to retrieve context, history,
    prepare the request, and send the request.
    """

    def __init__(
        self,
        aistudio_url: str,
        aistudio_key: str,
        az_monitor: Optional[str] = None,
    ) -> None:
        """
        Initialize the BaseGenerator with required parameters.

        Args:
            aistudio_url (str): The URL of the Azure AI Studio service.
            aistudio_key (str): The API key for accessing the Azure AI Studio service.
            az_monitor (Optional[str], optional): Connection string for Azure Monitor. Defaults to None.
        """
        self.aistudio_url = aistudio_url
        self.aistudio_key = aistudio_key
        self.__system_message: str = DEFAULT_SYSTEM_MESSAGE
        if az_monitor:
            logger.addHandler(AzureLogHandler(connection_string=az_monitor))
        self.waiting_time = 1
        self.http_client = httpx.AsyncClient(timeout=20000)

    async def __call__(
            self,
            prompt: str,
            streaming: bool = False,
            complete_response: bool = False,
            *args,
            **kwds
        ):
        """
        Template method to retrieve context and history, prepare the request, and send it.

        Args:
            streaming (bool, optional): Whether the response should be streamed. Defaults to False.

        Returns:
            The response from the AI Studio service.
        """
        context = await self.retrieve_context()
        history = await self.retrieve_history()
        prompt=Template(DEFAULT_PROMPT).safe_substitute(
            prompt=prompt,
            context=context,
            history=history
        )
        return await self.send_request(prompt, stream=streaming, complete_response=complete_response, *args, **kwds)

    async def _stream_url(self, url, method, data=None):
        """
        Stream data from the given URL using the specified HTTP method.

        Args:
            url (str): The URL to send the request to.
            method (str): The HTTP method to use (e.g., 'GET', 'POST').
            data (Optional[dict], optional): The data to include in the request. Defaults to None.

        Yields:
            str: Chunks of the response text.
        """
        request_param = {
            "method": method,
            "url": url,
            "headers": self.headers,
        }
        if data:
            request_param["json"] = data
        try:
            async with self.http_client.stream(**request_param) as response:
                self.waiting_time = 0
                response.raise_for_status()
                logger.info("Request Successful: %s", response.status_code)
                async for chunk in response.aiter_text():
                    yield chunk
        except httpx.NetworkError as exc:
            logger.error("Network Error: %s. Trying Again.", str(exc))
            await asyncio.sleep(0.5)
            yield self._stream_url(url, method, data)
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
            self.waiting_time += 1
            yield self._stream_url(url, method, data)

    async def _request_url(self, url, method, data=None):
        """
        Send a request to the given URL using the specified HTTP method and return the response.

        Args:
            url (str): The URL to send the request to.
            method (str): The HTTP method to use (e.g., 'GET', 'POST').
            data (Optional[dict], optional): The data to include in the request. Defaults to None.

        Returns:
            dict: The JSON response from the server.
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
            self.waiting_time += 1
            return await self._request_url(url, method, data)

    @property
    def headers(self):
        """
        Get the headers for the HTTP request.

        Returns:
            dict: The headers including the API key.
        """
        return {
            "Content-Type": "application/json",
            "api-key": self.aistudio_key,
        }

    @property
    def system_message(self) -> str:
        """
        Get the system message.

        Returns:
            str: The current system message.
        """
        return self.__system_message

    @system_message.setter
    def system_message(self, message: str) -> None:
        """
        Set a new system message.

        Args:
            message (str): The new system message.
        """
        self.__system_message = message

    @system_message.deleter
    def system_message(self) -> None:
        """
        Reset the system message to the default value.
        """
        self.__system_message = DEFAULT_SYSTEM_MESSAGE

    async def close(self) -> None:
        """
        Close the HTTP client.
        """
        await self.http_client.aclose()

    @abstractmethod
    async def retrieve_context(self) -> str:
        """
        Abstract method to retrieve the context.

        Subclasses should implement this method to retrieve the context based on specific requirements.

        Returns:
            str: The retrieved context.
        """

    @abstractmethod
    async def retrieve_history(self) -> str:
        """
        Abstract method to retrieve the history of the conversation.

        Subclasses should implement this method to retrieve the conversation history.

        Returns:
            str: The retrieved history.
        """

    @abstractmethod
    async def prepare_request(self, prompt_template: PromptTemplate) -> str:
        """
        Abstract method to prepare the request to Azure AI Studio.

        Constructs the query request based on given parameters.

        Args:
            prompt_template (PromptTemplate): The prompt text to generate the query.

        Returns:
            str: The prepared prompt request as a string.
        """

    @abstractmethod
    async def send_request(
        self,
        prompt: str,
        parameters: Dict[str, str | float | int],
        *,
        complete_response: bool = False,
        stream: bool = False
    ) -> dict:
        """
        Abstract method to send the request to the model endpoint.

        Args:
            prompt (str): The prepared prompt to be sent.
            parameters (Dict[str, str | float | int]): Additional parameters for the request.
            complete_response (bool, optional): Whether to return the complete response. Defaults to False.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            dict: The response from the model endpoint.
        """
