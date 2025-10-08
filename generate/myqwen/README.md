## 下载模型

```shell
python -m pip install "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen-Image --local-dir /root/workspace/T2I
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen-Image-Edit --local-dir /root/workspace/I2I
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen-Image-Edit-2509 --local-dir /root/workspace/MI2I
```

## 配置环境

```shell
pip install --upgrade git+https://github.com/huggingface/diffusers.git # qwen

pip install transformers
pip install accelerate
pip install --upgrade torch torchvision
```