"""
This Python module is designed to define data models and structures for interacting with Azure AI services, 
specifically tailored for generating and processing requests involving natural language processing. 
It utilizes Pydantic for model validation, ensuring data integrity and type safety.

The module defines three main classes, each representing a specific data structure required to construct and 
send requests to Azure AI services. These models encapsulate data for AI-driven operations, such as text generation, 
querying, or other AI tasks. They are particularly useful for applications integrating Azure AI functionalities.

Each class is defined using Pydantic, which adds automatic validation of the data fields, ensuring that the data 
passed to Azure AI services is in the correct format and adheres to the expected schema. This module is ideal for 
developers working on AI-powered applications requiring structured and validated input for Azure AI services.
"""


from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, field_validator, Field


CONTENT_TYPE = List[Dict[str, Union[str, Dict[str, str]]]]


class AzureAIMessage(BaseModel):
    """
    Data Specification for Using tools.
    DISCLAIMER: Should be enabled by the model.
    """
    role: Literal["system", "user"]
    content: CONTENT_TYPE


class AzureDateSource(BaseModel):
    """
    Data Specification for Using tools.
    DISCLAIMER: Should be enabled by the model.
    """
    type: str
    parameters: Dict[str, Any]


class AzureAIFunction(BaseModel):
    """
    Data Specification for Using tools.
    DISCLAIMER: Should be enabled by the model.
    """
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]


class AzureAITool(BaseModel):
    """
    Data Specification for Using tools.
    DISCLAIMER: Should be enabled by the model.
    """
    type: str
    function: AzureAIFunction


class AzureAIRequest(BaseModel):
    """
    Follows Azure AI Studio Swagger spec for different Models.
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
    dataSources: Optional[List[AzureDateSource]] = None
    seed: Optional[int] = None
    tools: Optional[List[AzureAITool]] = None

    @field_validator("stop")
    @classmethod
    def validate_stop_words(cls, value):
        if len(value) > 4:
            raise ValueError("The number of stop words should not exceed 4.")
        return value

    @field_validator("n")
    @classmethod
    def validate_max_completion(cls, value):
        if not value in list(range(1, 128)):
            raise ValueError("The maximum value of completions generated should be 128, while the minimum is 1.")
        return value


class PromptTemplate(BaseModel):
    prompt: Optional[str] = Field(..., description="The prompt used for the question.")
    function_name: Optional[str] = Field(..., description="The function name that should be applied on this workload.")
