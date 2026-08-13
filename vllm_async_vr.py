"""
    Veronica Rammouz 2026.

    This script demonstrates how to use the vLLM library to serve a model, evaluate a set of test cases, and print output(s) efficiently.
    Adapt as needed for your own evaluation tasks.
    Use with caution and prudence for appropriate resource management. Note the port number differences.

    Step 1: choose one of the following options to serve the model(s) using vLLM:
        Option A: Single model (check if no one else is running the same model on the same GPU)
            CUDA_VISIBLE_DEVICES=2 nohup vllm serve meta-llama/Llama-3.1-8B-Instruct \
            --host 127.0.0.1 --port 8002 --gpu-memory-utilization  0.9 > vllm.log 2>&1 & \

        Option B: Two models on the same GPU with 45% memory each. Less likely to work if the models are larger and the GPU has limited memory.
            CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
            --host 127.0.0.1 \
            --port 8000 \
            --gpu-memory-utilization 0.45

            CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
            --host 127.0.0.1 \
            --port 8001 \
            --gpu-memory-utilization 0.45

        Option C: Two models on different GPUs
            CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000
            CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8001

    Step 2: Run the evaluation script to test the model's performance on a set of test cases. This is where multiple users can run evaluations on the same model without interfering with each other, as long as they are using different ports.
        CUDA_VISIBLE_DEVICES=2 nohup python vllm_async_vr.py > eval.log 2>&1 &

    Note: common fix to frequent connection errors is changing the port number in the vllm serve command and in the evaluation script.
"""

import openai
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import asyncio
import aiohttp
import numpy as np


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
async def create_completion_prompt(session, messages: list, model_name, model_max_tokens) -> str:
    """
    Creates a completion prompt for the OpenAI API.
    Args:
        messages (list): The list of messages to include in the prompt.
    Returns:
        str: The generated prompt.
    """
    if isinstance(messages, dict):
        messages = [messages]
    semaphore = asyncio.Semaphore(1)

    client = openai.AsyncOpenAI(api_key="fake_key",base_url = "http://127.0.0.1:8000/v1",
                                timeout=300.0,)

    # n = 1  # consistency parameter i.e., number of completions to generate for each prompt

    for attempt in range(5):
        try:
            async with semaphore:
                # for _ in range(n): # <- consistency loop
                response = await client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            response_format={"type": "text"}, 
                            # suppress_cot=True,
                            max_tokens=model_max_tokens,
                            # temperature=model_temperature,
                            # top_p=model_top_p, 
                            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                            # logprobs=True,
                            # top_logprobs=6,
                            timeout=30
                        )
                try:
                    if response and response.choices and len(response.choices) > 0:
                        return response.choices[0].message.content.lower()
                    else:
                        raise ValueError("Response is missing choices.")
                except Exception as e:
                    print(f"Error processing response: {e}")
                    return 0  
        except (openai.APIConnectionError, openai.RateLimitError) as e:
            print(f"Connection error: {e}. Retrying in 2s...")
            time.sleep(2)
        raise Exception("Failed after 5 attempts")


async def evaluate(test_cases, model_id="meta-llama/Llama-3.1-8B-Instruct", max_tokens=24):
    """
    Evaluates each test case by sending the item query to the LLM.
    Args:
        test_cases (list): List of dictionaries containing user queries.
        model_id (str): The model identifier to use for the evaluation.
        max_tokens (int): The maximum number of tokens to generate in the response.
    Returns:
        None. Prints the responses for a test case.
    """

    responses = []

    messages = [
            (
                {"role" : "system", "content" : "You are a helpful assistant."},
                {"role" : "user", "content" : f"What is the capital of {item['country']}?"}
            )
        for item in test_cases
    ]

    # useful when data is large
    messages = np.array_split(messages, 30)

    for message_list in messages:
        async with aiohttp.ClientSession() as session:
            tasks = [create_completion_prompt(session, prompt, model_id, max_tokens) for prompt in message_list]
            responses = responses + await asyncio.gather(*tasks)

    print(f"Example response: {responses[0]}")


def main():
    """
        Main function to run the evaluation.
    """

    data = [
        {"country": "France", "capital": "Paris"},
        {"country": "Germany", "capital": "Berlin"},
        {"country": "Italy", "capital": "Rome"},
        {"country": "Spain", "capital": "Madrid"},
        {"country": "Portugal", "capital": "Lisbon"},
        {"country": "Netherlands", "capital": "Amsterdam"},
        {"country": "Belgium", "capital": "Brussels"},
        {"country": "Switzerland", "capital": "Bern"},
    ]

    asyncio.run(evaluate(data))


if __name__ == "__main__":
    main()