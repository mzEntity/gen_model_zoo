import os

from utils.utils import setup, read_file_to_dict, save_dict_to_file

import importlib


def main(main_config_file_path):
    cfg = read_file_to_dict(main_config_file_path)

    pipeline_name = cfg.pipeline_name
    
    model_config_file_path = cfg.model_config_file_path
    model_cfg = read_file_to_dict(model_config_file_path)
    
    try:
        generate_manage_module = importlib.import_module("generate")
        pipeline_class = getattr(generate_manage_module, pipeline_name)
    except Exception as e:
        raise NotImplementedError(f"UnImplemented pipeline: {pipeline_name}. {e}")
    
    pipeline = pipeline_class(**model_cfg)
        
    input_config_file_path = cfg.input_config_file_path
    input_cfg = read_file_to_dict(input_config_file_path)
    
    input_base_dir = input_cfg.input_base
    output_base_dir = input_cfg.output_base
    os.makedirs(output_base_dir, exist_ok=True)
    
    pipeline.set_io_base(input_base_dir, output_base_dir)
    
    save_dict_to_file(input_cfg, os.path.join(output_base_dir, "input.json"))
        
    for task_idx, task_item in enumerate(input_cfg['task_list']):        
        print(f"processing task {task_idx}...")
        generate_result = pipeline(**task_item['input'])
        pipeline.save(generate_result, *task_item['output'])
        print(f"result saved to local.")

    pipeline.close()
        

if __name__ == "__main__":
    print("Hello World!")
    setup()
    main("config/config.yaml")