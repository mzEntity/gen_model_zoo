import os

from utils.utils import setup, save_dict_to_json, read_json_to_dict

from generate.generating import (
    get_video_generate_func,
    release_video_pipeline
)

from generate.mywan.video_gen import WanTI2V_generate_video

GENERATE_MODE = "TI2V"

def main():
    ti2v_prompt_dict = read_json_to_dict("input/ti2v.json")
    print(ti2v_prompt_dict)
    
    output_dir = f"output/{GENERATE_MODE}"
        
    os.makedirs(output_dir, exist_ok=True)
    
    save_dict_to_json(ti2v_prompt_dict, os.path.join(output_dir, "input.json"))
    
    language = ti2v_prompt_dict['lang']

    cfg = {
        'model_path': '/root/workspace/TI2V'
    }

    video_gen_func = get_video_generate_func(cfg, WanTI2V_generate_video)
    
    for prompt_id, prompt_item in enumerate(ti2v_prompt_dict['prompt_list']):
        video_save_path = os.path.join(output_dir, f"{prompt_id}.mp4")
        prompt = prompt_item['prompt']
        
        reference_image_path = prompt_item['reference_image_path']
        
        print(f"{GENERATE_MODE} generation prompt: {prompt}")
        video_shot = video_gen_func(prompt=prompt, first_frame_path=reference_image_path, frame_num=121, size='1280*704', save_path=video_save_path)
        print(f"video save to {video_save_path}")

    release_video_pipeline()


if __name__ == "__main__":
    print("Hello World!")
    setup()
    main()