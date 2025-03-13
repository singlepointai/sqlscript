import json
import re
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

model_name = "/data/falcon3-7b-instruct"


def ask(n, prompt,
        display_output=False):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "min_tokens": 64,
        "max_tokens": 750,
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "stop": [],
        "skip_special_tokens": True,
        "spaces_between_special_tokens": False,
        'seed': 369
    }

    start_time = time.time()
    start_ftime = datetime.now().strftime('%H:%M:%S')
    response = requests.post(url, json=data, headers=headers)
    duration = time.time() - start_time
    end_ftime = datetime.now().strftime('%H:%M:%S')

    print(n, '-' * 60)
    print(f'Duration: {duration:.2f}s ({start_ftime} - {end_ftime}) | Status: {response.status_code}')

    completion_text = ""
    if response.status_code != 200:
        data = response.text
        print(f"!!! #: {n} Error: {response.text}")
    else:
        data = response.json()
        completion_text = data["choices"][0]["message"]["content"].strip()

        print(
            f'>>> Total Tokens: {data["usage"]["total_tokens"]} | Input: {data["usage"]["prompt_tokens"]} | Output: {data["usage"]["completion_tokens"]}')
        display_output and print(f'>>> Output: {data["choices"][0]["message"]["content"]}')

    # return format_model_response(completion_text)
    return completion_text


if __name__ == "__main__":
    start_time = time.time()


    content = json.load(open("input.json", "r", encoding="utf-8"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Create list of tasks in order
        tasks = [(i + 1, prompt) for i, prompt in enumerate(content)]

        # Map tasks to executor while maintaining order
        result = ""
        for res in executor.map(lambda x: ask(*x), tasks):
            result += res

    print(result)
    duration = time.time() - start_time

    print(f"Total time taken: {duration}")
