import pandas as pd
import argparse
import fire

from dataset import get_dataset, get_ref_dir_dataset, check_file_exists, format_question

from prompts import (
    image_to_text, 
    image_to_text_vqa_sam, 
    extract_segments_properties, 
    COT_PROMPT, 
    NO_COT_PROMPT,
    PROMPT_FOR_ANSWER
)

from models import (
    gpt4v_runner, 
    dalle3_runner, 
    semsam_runner, 
    fuyu_runner,
    gpt4v_openai_runner,
    gpt_chat_openai_runner,
)

from utils import check_dir, load_df, save_csv
from image_utils import image_to_base64, save_images

from typing import Optional


def encoder_runner(
    model_name: str = 'gpt-4v', 
    prompt: Optional[str] = '', 
    image: Optional[str] = None, 
    use_openai_api: bool = False,
    final_answer: bool = False,
):
    if model_name == 'gpt-4v':
        if not use_openai_api:
            return gpt4v_runner(prompt, image)
        else:
            if not image:
                model_name = 'gpt-4'
                return gpt_chat_openai_runner(model_name, prompt, final_answer=final_answer)
            else:
                return gpt4v_openai_runner(prompt, image)
    elif model_name.startswith('gpt-4') or model_name.startswith('gpt-3.5'):
        return gpt_chat_openai_runner(model_name, prompt, final_answer=final_answer)
    elif model_name == 'fuyu':
        return fuyu_runner(image)
    else:
        raise Exception('Invalid model name')

def decoder_runner(
    model_name: str = 'dall-e3', 
    prompt: Optional[str] = ''
):
    check_dir(f'experiments/{exp_dir}')
    
    if model_name == 'dall-e3':
        return dalle3_runner(prompt)
    # elif model_name == 'sdxl':
    #     return sdxl_runner(prompt)
    else:
        raise Exception('Invalid model name')

def run_img_to_img_experiments(
    exp_dir: str, 
    data_name: str = 'imagenet-1k-1000', 
    data_cap: int = 100, 
    prompt: str = '', 
    ref_dir: str = '', 
    use_sam: bool = False, 
    encoder_model: str = 'gpt-4v', 
    decoder_model: str = 'dall-e3', 
    use_ref_dir_responses: bool = False
) -> None:
    if not ref_dir:
        images_path = get_dataset(data_name, data_cap)
    else:
        images_path = get_ref_dir_dataset(data_name, ref_dir)
        print(f'images len: {len(images_path)}')
        
    existing_files = check_file_exists(exp_dir)
    print(f'existing files: {len(existing_files)}')
    df = load_df(f'experiments/{exp_dir}/', 'responses.csv') 
    print(f'df len: {df.image_path.nunique()}')
    
    for i, image_path in enumerate(images_path):
        print(f'processing image {i+1}/{len(images_path)}')
        
        df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
        filename = image_path.split('/')[-1]
        saved_images = False
        
        if filename in existing_files:
            print(f'{filename} already processed, skipping')
            continue
        else:
            print(f'now processing: {filename}')
        
        ref_df = load_df(f'experiments/{ref_dir}/', 'responses.csv')
        # if current image_path exists in ref_df['image_path'] set image_info to ref_df['gpt4-v response']
        if use_ref_dir_responses and filename in ref_df['image_path'].values:
            # re-use image_info extracted using encoder_model from ref_dir
            image_info = ref_df[ref_df['image_path'] == filename]['gpt4-v response'].values[0]
            print(f'found {filename} in ref_df, setting image_info to {image_info}')
            imageBase = image_to_base64(f'experiments/{ref_dir}/{filename}')
        else:
            if use_sam:
                imageBase = image_to_base64(image_path)
                saved_images = True
                imageBaseOriginal = imageBase
                imageBase = semsam_runner(imageBase)
                save_images([imageBase], f'/{exp_dir}', f"{filename.split('.')[0]}_sam.png", original=True)
            else:
                imageBase = image_to_base64(image_path)
            prompt = image_to_text if not len(prompt) else prompt
            image_info = encoder_runner(encoder_model, prompt, imageBase)
            print(image_info)
        
        curr_row = {'image_path': [filename], 'gpt4-v response': [image_info]}
        df = pd.concat([df, pd.DataFrame(curr_row)], ignore_index=True)
        save_csv(df, f'experiments/{exp_dir}/responses.csv')  
        
        output_base64 = decoder_runner(decoder_model, f'Recreate this image: {image_info}')
        if not saved_images:
            save_images([imageBase], f'/{exp_dir}', filename, original=True)
        else:
            save_images([imageBaseOriginal], f'/{exp_dir}', filename, original=True)
        save_images(output_base64, f'/{exp_dir}', filename.split('.')[0])

# Visual Reasoning
def run_vqa_experiments(
    exp_dir: str, 
    data_name: str = 'mmmu', 
    data_cap: int = 100, 
    prompt: str = '', 
    ref_dir: str = '', 
    use_sam: bool = False, 
    use_cot: bool = False,
    encoder_model: str = 'gpt-4v', 
    use_openai_api: bool = False,
) -> None: 
    check_dir(f'experiments/{exp_dir}')
    existing_files = check_file_exists(exp_dir, check_sam=False)
    print(f'existing files: {len(existing_files)}')
    df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
    print(f'df len: {df.image_path.nunique()}')
    existing_images_path = df['image_path'].values
    if not ref_dir:
        # if there is no ref directory to use, get the dataset and exclude the images already processed
        images_path = get_dataset(data_name, dataset_cap_max=True)
        images_path = [i.split('/')[-1] for i in images_path]
        data_cap = data_cap - len(existing_images_path) # update data_cap such that we get max data_cap images for the run
        images_path = [i for i in images_path if i not in existing_images_path] # exclude the images that have already been processed
        print(f'data_cap: {data_cap}')
        print(f'files to inlude len: {len(images_path)}')
        images_path = get_dataset(data_name, data_cap, files_to_include=images_path)
    else: # if a reference directory is provided, use the images from that directory (to test over the same set of images)
        images_path = get_ref_dir_dataset(data_name, ref_dir, check_sam=False)
        images_path = [i.split('/')[-1] for i in images_path]
        images_path = [f"{i.split('_sam.png')[0]}.png" for i in images_path]
        images_path = [i for i in images_path if i not in existing_images_path] # exclude the images that have already been processed
        images_path = [f"./datasets/{data_name}/{i}" for i in images_path]
    
    # images_path should only contain paths to image files 
    images_path = [i for i in images_path if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')]

    questions_df = load_df(f'datasets/{data_name}/', 'prompts.csv')
    
    for i, image_path in enumerate(images_path):
        print(f'processing image {i+1}/{len(images_path)}')
        
        df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
        filename = image_path.split('/')[-1]
        saved_images = False
        
        if filename in existing_files:
            print(f'{filename} already processed, skipping')
            continue
        else:
            print(f'now processing: {filename}')
        
        print(f'image_path: {image_path}')
        
        question = questions_df[questions_df['image_path'] == image_path[2:]]['question'].values[0]
        options = questions_df[questions_df['image_path'] == image_path[2:]]['options'].values[0]
        if not len(options):
            print(f'skipping {filename} as no options found')
            continue
        answer = questions_df[questions_df['image_path'] == image_path[2:]]['answer'].values[0]
        input_question = format_question(question, options)
        
        if use_sam:
            print('using SAM')
            imageBase = image_to_base64(image_path)
            saved_images = True
            imageBaseOriginal = imageBase
            if f"{filename.split('.')[0]}_sam.png" in existing_files:
                imageBase = image_to_base64(f'experiments/{exp_dir}/{filename.split(".")[0]}_sam.png')
            else:
                imageBase = semsam_runner(imageBase)
                save_images([imageBase], f'/{exp_dir}', f"{filename.split('.')[0]}_sam.png", original=True)
                
            prompt = image_to_text_vqa_sam if not len(prompt) else prompt
            segments_info = encoder_runner(
                encoder_model, 
                prompt, 
                imageBase, 
                use_openai_api
            ) # first_call
            
            prompt = f'{extract_segments_properties}\n{segments_info}\nSegments with properties:'
            segments_properties_info = encoder_runner(
                encoder_model, 
                prompt, 
                imageBaseOriginal,
                use_openai_api
            ) # second_call
            
            if use_cot:
                prompt = f'{segments_properties_info}\n{input_question}\n{COT_PROMPT}' 
                cot_answer = encoder_runner(
                    encoder_model, 
                    prompt, 
                    imageBaseOriginal,
                    use_openai_api) # third_call
                
                prompt = f'{prompt}\n{cot_answer}\n{PROMPT_FOR_ANSWER}'
                final_answer, parsed_ans = encoder_runner(
                    encoder_model, 
                    prompt,
                    use_openai_api=use_openai_api,
                    final_answer=True
                ) # fourth_call
                
                row_data = {
                    'image_path': [filename],
                    'question': [input_question],
                    'cot_answer': [cot_answer],
                    'final_answer': [final_answer],
                    'parsed_answer': [parsed_ans],
                    'correct_answer': [answer], 
                    'prompt': [prompt]
                }
            else:
                prompt = f'{segments_properties_info}\n{input_question}\n{NO_COT_PROMPT}'
                final_answer = encoder_runner(
                    encoder_model, 
                    prompt, 
                    imageBaseOriginal,
                    use_openai_api) # third_call
                
                prompt = f'{prompt}\n{final_answer}\n{PROMPT_FOR_ANSWER}'
                final_answer, parsed_ans = encoder_runner(
                    encoder_model, 
                    prompt,
                    use_openai_api=use_openai_api,
                    final_answer=True
                ) # fourth_call
                row_data = {
                    'image_path': [filename],
                    'question': [input_question],
                    'answer': [final_answer],
                    'final_answer': [final_answer],
                    'parsed_answer': [parsed_ans],
                    'correct_answer': [answer],
                    'prompt': [prompt]
                }
        else:
            print('not using SAM')
            imageBase = image_to_base64(image_path)
            if use_cot: # Baseline- COT, No SAM/Hierarchical Visual Processing
                prompt = f'{input_question}\n{COT_PROMPT}'
                cot_answer = encoder_runner(
                    encoder_model, 
                    prompt, 
                    imageBase,
                    use_openai_api) # first_call
                
                prompt = f'{prompt}\n{cot_answer}\n{PROMPT_FOR_ANSWER}'
                final_answer, parsed_ans = encoder_runner(
                    encoder_model, 
                    prompt,
                    use_openai_api=use_openai_api,
                    final_answer=True) # second_call
                  
                row_data = {'image_path': [filename], 'question': [input_question], 
                'cot_answer': [cot_answer], 'final_answer': [final_answer], 
                'parsed_answer': [parsed_ans],
                'correct_answer': [answer],
                'prompt': [prompt]}
            else: # Baseline- No COT, No SAM/Hierarchical Visual Processing
                prompt = f'{input_question}\n{NO_COT_PROMPT}:'
                intermediate_ans = encoder_runner(
                    encoder_model, 
                    prompt, 
                    imageBase,
                    use_openai_api
                ) # first_call
                
                prompt = f'{prompt}\n{intermediate_ans}\n{PROMPT_FOR_ANSWER}'
                final_answer, parsed_ans = encoder_runner(
                    encoder_model, 
                    prompt,
                    use_openai_api=use_openai_api,
                    final_answer=True
                ) # second_call
                row_data = {
                    'image_path': [filename],
                    'question': [input_question],
                    'answer': [final_answer],
                    'final_answer': [final_answer],
                    'parsed_answer': [parsed_ans],
                    'correct_answer': [answer],
                    'prompt': [prompt]
                }
        
        curr_row = {'image_path': [filename]} 
        curr_row.update(row_data)
        df = pd.concat([df, pd.DataFrame(curr_row)], ignore_index=True)
        save_csv(df, f'experiments/{exp_dir}/responses.csv')  

def run_img_to_img_experiments_test(
    exp_dir: str, 
    use_sam: bool = False
) -> None:
    check_dir(f'experiments/{exp_dir}')
    filename = 'out.jpg'
    imageBase64 = image_to_base64(filename) # test image as input
    if use_sam:
        output_base64 = semsam_runner(imageBase64)
        save_images([output_base64], f'/{exp_dir}', filename.split('.')[0])
    else:
        prompt = image_to_text
        image_info = gpt4v_runner(prompt, imageBase64)
        print(image_info)
        df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
        curr_row = {'image_path': [filename], 'gpt4-v response': [image_info]}
        df = pd.concat([df, pd.DataFrame(curr_row)], ignore_index=True)
        save_csv(df, f'experiments/{exp_dir}/responses.csv')  
        output_base64 = dalle3_runner(image_info)
        save_images([imageBase64], f'/{exp_dir}', filename, original=True)
        save_images(output_base64, f'/{exp_dir}', filename.split('.')[0])
        
def run_openai_gpt4v_vqa_test(
    exp_dir: str,
    use_sam: bool = False,
    use_openai_api: bool = False,
) -> None:
    check_dir(f'experiments/{exp_dir}')
    filename = 'out.jpg'
    imageBase64 = image_to_base64(filename) # test image as input
    if use_sam:
        output_base64 = semsam_runner(imageBase64)
        save_images([output_base64], f'/{exp_dir}', filename.split('.')[0])
    else:
        prompt = image_to_text_vqa_sam
        image_info = encoder_runner(
            model_name='gpt-4v', 
            prompt=prompt, 
            image=imageBase64, 
            use_openai_api=use_openai_api
        )
        print(image_info)
        test_qa = encoder_runner(
            model_name='gpt-4', 
            prompt=f"{image_info}. What is the man doing?", 
            use_openai_api=use_openai_api
        )
        print(test_qa)
        df = load_df(f'experiments/{exp_dir}/', 'responses.csv')
        curr_row = {'image_path': [filename], 'gpt4-v response': [image_info]}
        df = pd.concat([df, pd.DataFrame(curr_row)], ignore_index=True)
        save_csv(df, f'experiments/{exp_dir}/responses.csv')

if __name__ == '__main__':
    fire.Fire(
            {
                "img_to_img": run_img_to_img_experiments,
                "vqa": run_vqa_experiments,
                "img_to_img_test": run_img_to_img_experiments_test,
                "vqa_gpt4v_test": run_openai_gpt4v_vqa_test,
            }
        )