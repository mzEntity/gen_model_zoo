## 下载模型

```shell
python -m pip install "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /root/workspace/TI2V
```

## 配置环境

```shell
pip install easydict
pip install diffusers
pip install ftfy

pip install transformers
pip install einops
pip install decord

pip install librosa
pip install imageio
pip install opencv-python
pip install opencv-python-headless
pip install peft

pip uninstall flash-attn
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
ip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install imageio[ffmpeg]

pip install moviepy
```