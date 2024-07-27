DEFAULT_SYSTEM_MESSAGE = """
    You are a intelligent assistant.\n
    In your prompts, you will receive semantic requests to process.\n
    Your role is to understand what the user asks and rationalize over it.\n
    if a function is passed in the prompt, you should call it and return the result.\n
"""

DEFAULT_PROMPT = f"""
    Provided the following context:
    {30*"-"}
    $context
    {30*"-"}
    And the following history:
    $history
    {30*"-"}
    $prompt
"""
