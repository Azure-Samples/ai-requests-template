from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import logging

from opencensus.ext.azure.log_exporter import AzureLogHandler

from src.system_message import DEFAULT_SYSTEM_MESSAGE


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BaseGenerator(ABC):
    """
    _summary_: Abstract base class representing a query generator using Azure OpenAI Service.
    This class provides the foundational methods and properties for generating queries
    and handling responses from the Azure OpenAI Service.

    Applies the "Template Method" pattern on the prepare_request method
    """

    def __init__(
        self,
        aoai_url: str,
        aoai_key: str,
        az_monitor: Optional[str] = None,
    ) -> None:
        """
        Initialize the PromptGenerator with Azure OpenAI Service URL, access key, and optional
        Azure Monitor connection string for logging.

        Args:
            aoai_url (str): The URL endpoint for the Azure OpenAI Service.
            aoai_key (str): Access key for Azure OpenAI Service authentication.
            az_monitor (Optional[str]): Connection string for Azure Monitor, used for logging.
        """
        self.aoai_url = aoai_url
        self.aoai_key = aoai_key
        self.__system_message: str = DEFAULT_SYSTEM_MESSAGE
        if az_monitor:
            logger.addHandler(AzureLogHandler(connection_string=az_monitor))
        self.waiting_time = 1

    @property
    def headers(self):
        """
        Property that returns the standard headers required for Azure OpenAI Service requests,
        including Content-Type and Authorization headers.

        Returns:
            dict: A dictionary containing the necessary HTTP headers.
        """
        return {
            "Content-Type": "application/json",
            "api-key": self.aoai_key,
        }

    @property
    def system_message(self) -> str:
        """
        Property getter for the system message to be used in Azure OpenAI Service requests.

        Returns:
            str: The current system message.
        """
        return self.__system_message

    @system_message.setter
    def system_message(self, message: str) -> None:
        """
        Property setter for updating the system message.

        Args:
            message (str): The new system message to set.
        """
        self.__system_message = message

    @system_message.deleter
    def system_message(self):
        """
        Property deleter for resetting the system message to its default value.
        """
        self.__system_message = DEFAULT_SYSTEM_MESSAGE
