class MyBasePipeline:
    def __init__(self):
        pass
    
    def set_io_base(self, input_base_dir, output_base_dir): 
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        
    def save(self, call_result, **output_cfg):
        raise NotImplementedError("save(call_result, **output_cfg) not implemented in subclass")
    
    def close(self):
        raise NotImplementedError("close() not implemented in subclass")