from diffusers import QwenImagePipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline

import torch
from PIL import Image
import gc
import os

from generate.generating import MyBasePipeline

class MyQwenImagePipeline(MyBasePipeline):
    def __init__(self, model_path, language, ratio):
        self.model_path = model_path
        
        if not torch.cuda.is_available():
            raise ValueError("MyQwenImagePipeline require cuda, which is not available.")
        
        self.torch_dtype = torch.bfloat16
        self.device = "cuda"
            
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        if ratio not in aspect_ratios:
            ratio = "16:9"
            
        self.width, self.height = aspect_ratios[ratio]
        
        self.positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": ", 超清，4K，电影级构图." # for chinese prompt
        }
        
        if language not in self.positive_magic:
            raise ValueError(f"Unsupported language {language}")

        self.language = language

        self.pipeline = QwenImagePipeline.from_pretrained(self.model_path, torch_dtype=self.torch_dtype)
        self.pipeline = self.pipeline.to(self.device)
        
            
    def __call__(self, text, negative_text=' '):
        if self.pipeline is None:
            raise ValueError("The pipeline has already been closed.")
        image = self.pipeline(
            prompt = text + self.positive_magic[self.language],
            negative_prompt=negative_text,
            width=self.width,
            height=self.height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]
        return image
    
    
    def save(self, image, **output_cfg):
        save_path = output_cfg['path']
        image.save(os.path.join(self.output_base_dir, save_path))
    
    
    def close(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        


class MyQwenImageEditPlusPipeline(MyBasePipeline):
    def __init__(self, model_path, language):
        self.model_path = model_path
        
        if not torch.cuda.is_available():
            raise ValueError("MyQwenImageEditPlusPipeline require cuda, which is not available.")
        
        self.torch_dtype = torch.bfloat16
        self.device = "cuda"
                
        if language != 'en':
            raise ValueError(f"Unsupported language {language}")

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(self.model_path, torch_dtype=self.torch_dtype)
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=None)
        
        
    def __call__(self, text, image_path_list, negative_text=' '):
        if self.pipeline is None:
            raise ValueError("The pipeline has already been closed.")
        
        image_list = []
        for image_path in image_path_list:
            image = Image.open(os.path.join(self.input_base_dir, image_path))
            image_list.append(image)
            
        inputs = {
            "image": image_list,
            "prompt": text,
            "generator": torch.manual_seed(42),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }
        
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]
            return output_image
    
    
    def save(self, image, **output_cfg):
        save_path = output_cfg['path']
        image.save(os.path.join(self.output_base_dir, save_path))
    
    
    def close(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()

