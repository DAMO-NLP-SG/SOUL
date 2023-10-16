import os
import json
import re
import openai
import pandas as pd
import argparse
import random
import requests
from tqdm import tqdm
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import concurrent.futures

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Define model URLs for Hugging Face models
model_urls = {
    "flant5": "https://api-inference.huggingface.co/models/google/flan-t5-xxl",
    "flanul2": "https://api-inference.huggingface.co/models/google/flan-ul2"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, required=True, help="input data path")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of samples to use, better under 3")
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot]")
    parser.add_argument("--nrows", type=int, default=None, help="number of rows")
    parser.add_argument("--api", type=str, default=None, help="api key")
    parser.add_argument("--model", type=str, default="chat", help="[chat, flant5, flanul2]")
    parser.add_argument("--skip_runned", action="store_true", help="skip runned rows")
    parser.add_argument("--use_justification", action="store_true", help="true - jg task; false - rc task")
    return parser.parse_args()


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

@retry(wait=wait_fixed(3), stop=stop_after_attempt(6), before=before_retry_fn)
def query_model(model_type: str, api_key: str, prompt: str, max_tokens: int = 256, temperature: float = 0) -> str:
    # OpenAI ChatGPT and DaVinci models
    if model_type in ["chat", "davinci"]:
        openai.api_key = api_key
        try:
            if model_type == "davinci":
                completions = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                output = completions.choices[0].text.strip()
            elif model_type == "chat":
                completions = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                    temperature=temperature,
                )
                output = completions.choices[0].message.content.strip()
        except Exception as e:
            print(e)
        return output

    # Hugging Face Flan-T5 and Flan-UL2 models
    elif model_type in ["flant5", "flanul2"]:
        model_url = model_urls[model_type]
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "max_new_tokens": max_tokens,
            }
        }
        try:
            response = requests.post(model_url, headers=headers, json=payload)
            pred = response.json()[0]['generated_text'].strip()
        except Exception as e:
            error = response.json()
            print(error)
            if "must have less than" in error["error"]:
                print("Too long, regard as error")
                return "error"
        return pred

    else:
        raise ValueError(f"Invalid model type: {model_type}. Expected 'chat', 'davinci', 'flant5', or 'flanul2'.")


def parallel_query_model(kwargs) -> str:
    return query_model(**kwargs)


def generate_prompt(setting, review, state, use_just=False):

    if setting == "zero-shot":
        if use_just:
            prompt = "Given the review below:\n\n{review}\n\nAnswer the following two questions without missing any one:\n1. Is the following statement true, false or not-given; \n2. What is your reason? \n\n{state}".format(review=review, state=state)
        else:
            prompt_template = "Given the review below:\n\n{review}\n\nIs the following statement true, false or not-given?\n\n{state}"
            prompt = prompt_template.format(review=review, state=state)
            prompt += "\n\nReturn your answer in lower case without any other text."
    else:
        raise NotImplementedError(f"Invalid setting, selected from ['zero-shot']")

    return prompt


def main():
    args = parse_args()

    # prepare output folder
    nrows = f"_{args.nrows}" if args.nrows else ""
    subtask = f"jg" if args.use_justification else "rc"
    output_folder = f"./output/{args.setting}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"pred{nrows}_{args.model}_{args.setting}_{subtask}.csv")

    read_path = output_path if (os.path.exists(output_path) and args.skip_runned) else args.data_path
    df = pd.read_csv(read_path, nrows=args.nrows)

    df["prediction"] = df.get("prediction", "")
    df["prompt"] = df.get("prompt", "")
    if args.use_justification:
        df["output"] = df.get("output", "")

    prompt_args = []
    prompts = []

    start_idx = 0

    for index, row in df.iterrows():
        # find the start point
        if args.skip_runned:
            if not (pd.isnull(row["prediction"]) or row["prediction"] == ""):
                start_idx += 1
                continue

        prompt = generate_prompt(setting=args.setting, review=row["review_text"],
                                state=row["statement"], use_just=args.use_justification)
        prompt_args.append({
            "model_type": args.model,
            "api_key": args.api,
            "prompt": prompt,
        })
        prompts.append(prompt)

    assert start_idx + len(prompts) == len(df)
    print('='*100)
    print(prompts[0])
    print('='*100)

    print("df length:", len(df))

    if start_idx != 0:
        print(f"Continue from {start_idx}/{len(df)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        predictions = []
        for prediction in tqdm(executor.map(parallel_query_model, prompt_args), total=len(prompt_args), desc=f"Calling", dynamic_ncols=True):
            predictions.append(prediction)
            idx = len(predictions)-1+start_idx
            if args.use_justification:
                df.loc[idx, "output"] = prediction
                pattern = re.compile("(true|false|not-given)", re.IGNORECASE)
                try:
                    text = prediction.lower().strip().replace("not given", "not-given")
                    pred = re.findall(pattern, text)[0]
                except:
                    pred = "error"
                df.loc[idx, "prediction"] = pred
            else:
                df.loc[idx, "prediction"] = prediction
            df.loc[idx, "prompt"] = prompts[len(predictions)-1]
            # temporary save
            if (idx + 1) % 50 == 0 or idx == len(prompt_args) - 1:
                df.to_csv(output_path, index=False)
    # just to make sure the final save
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

