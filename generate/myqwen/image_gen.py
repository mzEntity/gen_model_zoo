from diffusers import QwenImagePipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline
import torch
from PIL import Image
import gc


class MyQwenImagePipeline:
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
        
        
    def __call__(self, prompt, negative_prompt=' '):
        if self.pipeline is None:
            raise ValueError("The pipeline has already been closed.")
        image = self.pipeline(
            prompt = prompt + self.positive_magic[self.language],
            negative_prompt=negative_prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]
        return image
    
    
    def close(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        


def QwenImage_generate_image(cfg):
    pipeline = MyQwenImagePipeline(cfg['model_path'], cfg['language'], cfg['ratio'])

    def gen(**kwargs):
        prompt = kwargs.get("prompt", "")
        return pipeline(prompt)
        
    gen._pipeline = pipeline
    return gen



class MyQwenImageEditPlusPipeline:
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
        
        
    def __call__(self, prompt, image_path_list, negative_prompt=' '):
        if self.pipeline is None:
            raise ValueError("The pipeline has already been closed.")
        
        image_list = []
        for image_path in image_path_list:
            image = Image.open(image_path)
            image_list.append(image)
            
        inputs = {
            "image": image_list,
            "prompt": prompt,
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
    
    
    def close(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        
        
def QwenImageEditPlus_generate_image(cfg):
    pipeline = MyQwenImageEditPlusPipeline(cfg['model_path'], cfg['language'])

    def gen(**kwargs):
        prompt = kwargs.get("prompt", "")
        image_path_list = kwargs.get("image_path_list", [])
        return pipeline(prompt, image_path_list)
        
    gen._pipeline = pipeline
    return gen


