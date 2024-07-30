import pytest
from aistudio_requests.schemas import PromptTemplate, AzureAIFunction


class TestConcreteFunctionCallingGenerator:

    @pytest.mark.asyncio
    async def test_function_from_class(self, function_calling):
        functions = [AzureAIFunction.from_python_function(function_calling.sum), ]
        assert isinstance(functions, list)
        for func in functions:
            assert isinstance(func, AzureAIFunction)

    @pytest.mark.asyncio
    async def test_simple_prompt(self, function_calling):
        prompt_template = PromptTemplate(prompt="How much is 2 + 2?")
        functions = [AzureAIFunction.from_python_function(function_calling.sum), ]
        parameters = {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 2000,
        }
        function_calling.functions = functions
        result = await function_calling(prompt_template, parameters=parameters)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_complete_prompt(self, function_calling):
        prompt_template = PromptTemplate(prompt="How much is 2 + 2?")
        functions = [AzureAIFunction.from_python_function(function_calling.sum), ]
        parameters = {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 2000,
        }
        function_calling.functions = functions
        result = await function_calling(prompt_template, complete_response=True, parameters=parameters)
        assert isinstance(result, dict)
