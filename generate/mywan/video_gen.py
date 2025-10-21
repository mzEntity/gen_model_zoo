# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import sys
import warnings

import gc

warnings.filterwarnings('ignore')

import torch
from PIL import Image

import generate.mywan.wan as wan
from generate.mywan.wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from generate.mywan.wan.utils.utils import save_video

from generate.generating import MyBasePipeline


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


class MyWanTI2VPipeline(MyBasePipeline):
    def __init__(self, model_path):
        _init_logging()
        cfg = WAN_CONFIGS['ti2v-5B']
        logging.info(f"Generation model config: {cfg}")
        
        rank = 0    
        device_id = 0
        t5_cpu = True
        convert_model_dtype = True
        
        logging.info("Creating WanTI2V pipeline.")
        self.pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=device_id,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )
    
    def __call__(self, text, image_path, frame_num=121, size='1280*704'):
        size_choices = ('704*1280', '1280*704')
        
        assert size in size_choices
        assert (frame_num - 1) % 4 == 0 # 4n+1
            
        seed = 42
                
        logging.info(f"Input prompt: {text}")

        img = Image.open(image_path).convert("RGB")
        logging.info(f"Input image: {image_path}")
        
        logging.info(f"Generating video ...")
        sample_guide_scale = 5.0
        sample_steps = 50
        offload_model = True
        sample_solver = 'unipc'
        sample_shift = 5.0
        video = self.pipeline.generate(
            text,
            img=img,
            size=SIZE_CONFIGS[size],
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=sample_guide_scale,
            seed=seed,
            offload_model=offload_model)
        
        logging.info("Finished.")
        return video
    
    def save(self, video, **output_cfg):
        save_path = output_cfg['path']
        sample_fps = 24
        save_video(
            tensor=video[None],
            save_file=os.path.join(self.output_base_dir, save_path),
            fps=sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    
    def close(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
    