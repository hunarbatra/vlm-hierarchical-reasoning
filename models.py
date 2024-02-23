import tenacity

import requests
import json
import functools

from openai import OpenAI

from typing import Optional

from image_utils import image_to_base64, save_images
from utils import extract_answer
from prompts import EXTRACT_ANSWER_GAURDRAIL_PROMPT

from gradio_client import Client

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
org_id = os.getenv('OPENAI_ORG_ID')
client = OpenAI(api_key=api_key, organization=org_id)

gpt4v_api_url = 'http://localhost:3000/'

headers = {
    'Content-Type': 'application/json',
}

fuyu_client = Client("https://adept-fuyu-8b-demo.hf.space/--replicas/pv8qh/") # might need to update this regularly
# sdxl_client = Client("https://hysts-sd-xl.hf.space/")

@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(120))
def gpt4v_runner(prompt: str, image: Optional[str] = None, model_name: str = 'gpt-4v'):
    data = {
        'prompt': prompt,
        'image': image,
    }
    response = requests.post(f'{gpt4v_api_url}{model_name}', headers=headers, data=json.dumps(data))
    response = response.json()
    return response["output"]

@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(120))
def dalle3_runner(prompt: str, model_name: str = 'dall-e3'):
    data = {
        'prompt': prompt,
    }
    response = requests.post(f'{gpt4v_api_url}{model_name}', headers=headers, data=json.dumps(data))
    response = response.json()
    return response["images"]

@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(10))
def semsam_runner(image: str):
    data = {
        'imageBase64': image,
    }
    response = requests.post(f'{gpt4v_api_url}semantic-sam', headers=headers, data=json.dumps(data))
    response = response.json()
    return response["image"]

# @tenacity.retry(stop=tenacity.stop_after_attempt(20), wait=tenacity.wait_fixed(20))
# def sdxl_runner(prompt: str):
#     print('starting sdxl run')
#     result = sdxl_client.predict(
# 		prompt,	# str  in 'Prompt' Textbox component
# 		"",	# str  in 'Negative prompt' Textbox component
# 		"",	# str  in 'Prompt 2' Textbox component
# 		"",	# str  in 'Negative prompt 2' Textbox component
# 		False,	# bool  in 'Use negative prompt' Checkbox component
# 		False,	# bool  in 'Use prompt 2' Checkbox component
# 		False,	# bool  in 'Use negative prompt 2' Checkbox component
# 		42,	# int | float (numeric value between 0 and 2147483647) in 'Seed' Slider component
# 		1024,	# int | float (numeric value between 256 and 1024) in 'Width' Slider component
# 		1024,	# int | float (numeric value between 256 and 1024) in 'Height' Slider component
# 		7.5,	# int | float (numeric value between 1 and 20) in 'Guidance scale for base' Slider component
# 		7.5,	# int | float (numeric value between 1 and 20) in 'Guidance scale for refiner' Slider component
# 		30,	# int | float (numeric value between 10 and 100) in 'Number of inference steps for base' Slider component
# 		30,	# int | float (numeric value between 10 and 100) in 'Number of inference steps for refiner' Slider component
# 		False,	# bool  in 'Apply refiner' Checkbox component
# 		api_name="/run"
#     )
#     print('sdxl run complete')
#     imgBase64 = image_to_base64(result)
#     return [imgBase64]
    
@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(10))
def fuyu_runner(image: str = None):
    if image:
        save_images([image], exp_dir='', filename='fuyu_img.png', original=True)
    response = fuyu_client.predict(
        "experiments/fuyu_img.png",	# str (filepath on your computer (or URL) of image) in 'Upload your Image' Image component
        True,	# bool  in 'Enable detailed captioning' Checkbox component
        fn_index=2
    )
    return response

def gen_openai_content(
    role: str = "user", 
    prompt: str = "", 
    image: Optional[str] = None
) -> dict:
    if image:
        return {"role": role, "content": gen_openai_image_content(prompt, image)}
    else:
        return {"role": role, "content": prompt}

def gen_openai_image_content(prompt: str, image_base64: str) -> list:
    if image_base64.startswith('data:image'):
        base64_data = image_base64.split(',', 1)[1]
    else:
        base64_data = image_base64
    return [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_data}"
        }
    ]
    
def validate_response_decorator(func):
    @functools.wraps(func)
    def wrapper_validate_response(*args, **kwargs):
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            # Run the original function
            ans = func(*args, **kwargs)
            # Validate the response
            validation_prompt = f'Validate if the model was able to generate a response for a given image successfully or not. Return "Y" or "N" only. Models response-- {ans}'
            validation_response = gpt_chat_openai_runner(model_name='gpt-3.5-turbo', prompt=validation_prompt, max_tokens=50)
            
            if validation_response == 'Y':
                return ans  # If validated successfully, return the answer
            attempt += 1  # Increase the attempt counter if validation failed
        
        # Return the last attempt's answer if all attempts fail
        return ans
    return wrapper_validate_response
    
# @validate_response_decorator
def gpt4v_openai_runner(
    prompt: str, 
    image: str,
    max_tokens: int = 512):
    messages = []
    messages.append(gen_openai_content("user", prompt, image))
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers_openai = headers.copy()
    headers_openai["Authorization"] = f"Bearer {api_key}"
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers_openai, json=payload)
    try:
        ans = response.json()['choices'][0]['message']['content']
    except:
        ans = response.json()
        print('err')
        print(image[:150])
        print(ans)
    return ans

def gpt_chat_openai_runner(
    model_name: str = "gpt-4",
    prompt: str = "",
    max_tokens: int = 512,
    final_answer: bool = False
):
    messages = []
    messages.append(gen_openai_content("user", prompt, ""))
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers_openai = headers.copy()
    headers_openai["Authorization"] = f"Bearer {api_key}"
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers_openai, json=payload)
    ans = response.json()['choices'][0]['message']['content']
    
    if final_answer:
        parsed_ans = extract_answer(ans)
        if parsed_ans is None:
            prompt = f'{EXTRACT_ANSWER_GAURDRAIL_PROMPT}\n{prompt}'
            final_answer = gpt_chat_openai_runner(model_name='gpt-3.5-turbo', prompt=prompt, max_tokens=128)
            parsed_ans = extract_answer(final_answer)
            parsed_ans = final_answer[0] if parsed_ans is None else parsed_ans
        return ans, parsed_ans
    
    return ans

