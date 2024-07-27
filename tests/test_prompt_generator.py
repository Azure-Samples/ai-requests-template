import pytest
from aistudio_requests.schemas import PromptTemplate


class TestConcretePromptGenerator:

    @pytest.mark.asyncio
    async def test_simple_prompt(self, prompt_generator):
        prompt_template = PromptTemplate(prompt="Test Template")
        parameters = {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 2000,
        }
        result = await prompt_generator(template=prompt_template, parameters=parameters)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_complete_prompt(self, prompt_generator):
        prompt_template = PromptTemplate(prompt="Test Template")
        parameters = {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 2000,
        }
        result = await prompt_generator(template=prompt_template, complete_response=True, parameters=parameters)
        assert isinstance(result, dict)
