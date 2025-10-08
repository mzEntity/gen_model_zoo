import os

from utils.utils import setup, save_dict_to_json, read_json_to_dict

from generate.generating import (
    get_image_generate_func,
    release_image_pipeline
)

from generate.myqwen.image_gen import QwenImage_generate_image


GENERATE_MODE = "T2I"

def main():
    t2i_prompt_dict = read_json_to_dict("input/t2i.json")
    print(t2i_prompt_dict)
    
    output_dir = f"output/{GENERATE_MODE}"
        
    os.makedirs(output_dir, exist_ok=True)
    
    save_dict_to_json(t2i_prompt_dict, os.path.join(output_dir, "input.json"))
    
    language = t2i_prompt_dict['lang']

    cfg = {
        'model_path': '/root/workspace/T2I',
        'language': language,
        'ratio': '16:9'
    }

    image_gen_func = get_image_generate_func(cfg, QwenImage_generate_image)
    
    for prompt_id, prompt_item in enumerate(t2i_prompt_dict['prompt_list']):
        img_save_path = os.path.join(output_dir, f"{prompt_id}.png")
        prompt = prompt_item['prompt']
        
        print(f"{GENERATE_MODE} generation prompt: {prompt}")
        img = image_gen_func(prompt=prompt)
        img.save(img_save_path)
        print(f"image save to {img_save_path}")

    release_image_pipeline()
        

if __name__ == "__main__":
    print("Hello World!")
    setup()
    main()