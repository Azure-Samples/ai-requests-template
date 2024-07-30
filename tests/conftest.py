import os
from string import Template
import pytest
from dotenv import load_dotenv

from aistudio_requests.generate import PromptGenerator, FunctionCallingGenerator
from aistudio_requests.schemas import PromptTemplate


CURRENT_DIR = os.path.dirname(__file__)
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

load_dotenv(dotenv_path)

url: str = os.environ.get("GPT4O_URL", "")
key: str = os.environ.get("GPT4O_KEY", "")
az_monitor: str = os.environ.get("AZ_CONNECTION_LOG", "")


# Concrete subclass for testing purposes
class ConcretePromptGenerator(PromptGenerator):

    async def retrieve_history(self):
        return ""

    async def retrieve_context(self):
        return ""

    async def prepare_request(self, prompt_template: PromptTemplate):
        return Template(self.prompt_template).safe_substitute(**prompt_template.model_dump())


# Concrete subclass for testing purposes
class ConcreteFunctionCallingGenerator(FunctionCallingGenerator):

    async def retrieve_history(self):
        return ""

    async def retrieve_context(self):
        return ""

    async def prepare_request(self, prompt_template: PromptTemplate):
        return Template(self.prompt_template).safe_substitute(**prompt_template.model_dump())

    def sum(self, a: int, b: int):
        return a + b


# Concrete subclass for testing purposes
class HistoryGenerator(FunctionCallingGenerator):

    def __init__(self, chat_id: str, **kwargs):
        super().__init__(**kwargs)
        self.chat_id = chat_id
        self.database_connection = []

    async def retrieve_history(self):
        return ""

    async def retrieve_context(self):
        return ""

    async def prepare_request(self, prompt_template: PromptTemplate):
        return Template(self.prompt_template).safe_substitute(**prompt_template.model_dump())

    async def execute_query(self, query: str):
        return f"{query} executed successfully"

# Concrete subclass for testing purposes
class ContextGenerator(PromptGenerator):

    def __init__(self, chat_id: str, **kwargs):
        super().__init__(**kwargs)
        self.chat_id = chat_id
        self.database_connection = []

    async def retrieve_history(self):
        return ""

    async def retrieve_context(self):
        return ""

    async def prepare_request(self, prompt_template: PromptTemplate):
        return ""


@pytest.fixture
def prompt_generator():
    return ConcretePromptGenerator(url, key, az_monitor)


@pytest.fixture
def function_calling():
    return ConcreteFunctionCallingGenerator(url, key, az_monitor)


@pytest.fixture
def history_generator():
    return HistoryGenerator(chat_id="test_chat_id")


@pytest.fixture
def context_generator():
    return ContextGenerator(chat_id="test_chat_id")
