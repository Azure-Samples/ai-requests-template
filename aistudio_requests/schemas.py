"""
This Python module defines data models and structures for interacting with Azure AI services, 
specifically tailored for generating and processing requests involving natural language processing. 
It utilizes Pydantic for model validation, ensuring data integrity and type safety.

The module defines several main classes, each representing a specific data structure required to construct and 
send requests to Azure AI services. These models encapsulate data for AI-driven operations, such as text generation, 
querying, or other AI tasks. They are particularly useful for applications integrating Azure AI functionalities.

Each class is defined using Pydantic, which adds automatic validation of the data fields, ensuring that the data 
passed to Azure AI services is in the correct format and adheres to the expected schema. This module is ideal for 
developers working on AI-powered applications requiring structured and validated input for Azure AI services.

Classes:
- AzureAIMessage: Defines the structure for messages to be consumed by language models.
- AzureDataSource: Defines the structure for specifying data sources.
- AzureAIFunction: Defines the structure for function calls.
- AzureAITool: Defines the structure for using tools.
- AzureAIRequest: Defines the overall request structure following Azure AI Studio Swagger specifications.
- PromptTemplate: Defines the template for prompts, including optional function names.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

CONTENT_TYPE = List[Dict[str, Union[str, Dict[str, str]]]]


class AzureAIMessage(BaseModel):
    """
    Data specification for writing messages consumable by language models.

    Attributes:
        role (Literal["system", "user"]): The role of the message sender. It can be either "system" or "user".
        content (CONTENT_TYPE): The content of the message, which is a list of dictionaries containing
                                strings or nested dictionaries.
    """

    role: Literal["system", "user"]
    content: CONTENT_TYPE


class AzureDataSource(BaseModel):
    """
    Data specification for using data sources in requests to Azure AI services.

    DISCLAIMER: Should be enabled by the model.

    Attributes:
        type (str): The type of the data source.
        parameters (Dict[str, Any]): Parameters required to access or use the data source.
    """

    type: str
    parameters: Dict[str, Any]


class AzureAIFunction(BaseModel):
    """
    Data specification for defining function calls in requests to Azure AI services.

    DISCLAIMER: Should be enabled by the model.

    Attributes:
        name (str): The name of the function.
        description (Optional[str]): An optional description of the function.
        parameters (Dict[str, Any]): Parameters required to call the function.
    """

    name: str
    description: Optional[str]
    parameters: Dict[str, Any]


class AzureAITool(BaseModel):
    """
    Data specification for using tools in requests to Azure AI services.

    DISCLAIMER: Should be enabled by the model.

    Attributes:
        type (str): The type of the tool.
        function (AzureAIFunction): The function associated with the tool.
    """

    type: str
    function: AzureAIFunction


class AzureAIRequest(BaseModel):
    """
    Represents a request to Azure AI services following Azure AI Studio Swagger specifications.

    Attributes:
        messages (List[AzureAIMessage]): List of messages to be processed by the AI model.
        temperature (Optional[float]): Sampling temperature for the model. Defaults to None.
        top_p (Optional[float]): Nucleus sampling probability. Defaults to None.
        n (Optional[int]): Number of completions to generate. Defaults to None.
        user (Optional[str]): Identifier for the user making the request. Defaults to None.
        max_tokens (int): Maximum number of tokens in the response. Defaults to 4096.
        stream (bool): Whether to stream the response. Defaults to False.
        presence_penalty (Optional[float]): Penalty for repeated tokens in the response. Defaults to None.
        frequency_penalty (Optional[float]): Penalty for frequency of tokens in the response. Defaults to None.
        stop (Optional[List[str]]): List of stop sequences. Defaults to None.
        logit_bias (Optional[Dict[str, float]]): Biases for specific tokens. Defaults to None.
        response_format (Literal["text", "json_object"]): Format of the response. Defaults to "text".
        dataSources (Optional[List[AzureDataSource]]): List of data sources to use. Defaults to None.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        tools (Optional[List[AzureAITool]]): List of tools to use. Defaults to None.
    """

    messages: List[AzureAIMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    user: Optional[str] = None
    max_tokens: int = 4096
    stream: bool = False
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    logit_bias: Optional[Dict[str, float]] = None
    response_format: Literal["text", "json_object"] = "text"
    dataSources: Optional[List[AzureDataSource]] = None
    seed: Optional[int] = None
    tools: Optional[List[AzureAITool]] = None

    @field_validator("stop")
    @classmethod
    def validate_stop_words(cls, value):
        """
        Validate the 'stop' field to ensure it contains no more than 4 stop words.

        Args:
            value (List[str]): The list of stop words.

        Returns:
            List[str]: The validated list of stop words.

        Raises:
            ValueError: If the number of stop words exceeds 4.
        """
        if len(value) > 4:
            raise ValueError("The number of stop words should not exceed 4.")
        return value

    @field_validator("n")
    @classmethod
    def validate_max_completion(cls, value):
        """
        Validate the 'n' field to ensure it is within the acceptable range.

        Args:
            value (int): The number of completions to generate.

        Returns:
            int: The validated number of completions.

        Raises:
            ValueError: If the value is not between 1 and 128 inclusive.
        """
        if not value in list(range(1, 128)):
            raise ValueError(
                "The maximum value of completions generated should be 128, while the minimum is 1."
            )
        return value


class PromptTemplate(BaseModel):
    """
    Represents a template for prompts used in Azure AI service requests.

    Attributes:
        prompt (str): The prompt text for the query.
        function_name (Optional[str]): The name of the function to be applied to this workload. Defaults to None.
    """

    prompt: str = Field(..., description="The prompt used for the question.")
    history: Optional[str] = Field(default=None, description="The chat history, if Any.")
    context: Optional[str] = Field(default=None, description="The context for the request, if Any.")
    function_name: Optional[str] = Field(
        default=None, description="The function name that should be applied on this workload."
    )
