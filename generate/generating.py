class DemoImagePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, prompt):
        print(f"Generate an image with prompt: '{prompt}'")
        
    def close(self):
        print(f"Releasing ImagePipeline resources...")
        
        
class DemoVideoPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, prompt):
        print(f"Generate a video with prompt: '{prompt}'")
        
    def close(self):
        print(f"Releasing VideoPipeline resources...")
        

class DemoVoicePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, prompt):
        print(f"Generate a voice with prompt: '{prompt}'")
        
    def close(self):
        print(f"Releasing VoicePipeline resources...")


def demo_generate_image(cfg):
    pipeline = DemoImagePipeline(cfg)

    def gen(**kwargs):
        prompt = kwargs.get("prompt", "")
        return pipeline(prompt)

    # 绑定 pipeline 到函数对象，方便释放
    gen._pipeline = pipeline
    return gen


def demo_generate_video(cfg):  
    pipeline = DemoVideoPipeline(cfg)

    def gen(**kwargs):
        prompt = kwargs.get("prompt", "")
        return pipeline(prompt)

    gen._pipeline = pipeline
    return gen


def demo_generate_voice(cfg):  
    pipeline = DemoVoicePipeline(cfg)

    def gen(**kwargs):
        prompt = kwargs.get("prompt", "")
        return pipeline(prompt)

    gen._pipeline = pipeline
    return gen


# --- 单例缓存 ---
_image_generate_func = None
_video_generate_func = None
_voice_generate_func = None


def get_image_generate_func(cfg, init_func):
    global _image_generate_func
    if _image_generate_func is None:
        _image_generate_func = init_func(cfg)
    return _image_generate_func


def get_video_generate_func(cfg, init_func):
    global _video_generate_func
    if _video_generate_func is None:
        _video_generate_func = init_func(cfg)
    return _video_generate_func


def get_voice_generate_func(cfg, init_func):
    global _voice_generate_func
    if _voice_generate_func is None:
        _voice_generate_func = init_func(cfg)
    return _voice_generate_func


def release_image_pipeline():
    """释放图片 pipeline 占用的显存"""
    global _image_generate_func
    if _image_generate_func is not None:
        _image_generate_func._pipeline.close()
        _image_generate_func = None


def release_video_pipeline():
    """释放视频 pipeline 占用的显存"""
    global _video_generate_func
    if _video_generate_func is not None:
        _video_generate_func._pipeline.close()
        _video_generate_func = None
        
        
def release_voice_pipeline():
    """释放视频 pipeline 占用的显存"""
    global _voice_generate_func
    if _voice_generate_func is not None:
        _voice_generate_func._pipeline.close()
        _voice_generate_func = None