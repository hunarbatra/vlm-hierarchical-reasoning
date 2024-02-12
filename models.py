import tenacity

import requests
import json

from typing import Optional

from image_utils import image_to_base64, save_images
from gradio_client import Client

gpt4v_api_url = 'http://localhost:3000/'
headers = {
    'Content-Type': 'application/json',
}

fuyu_client = Client("https://adept-fuyu-8b-demo.hf.space/--replicas/qlh1n/") # might need to update this regularly
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
    



