import os

# 1. 【核心】强制设置 HF 国内镜像站
# 这行代码会让所有下载请求走国内节点，速度极快
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
from datasets import load_dataset

print("🚀 开始通过国内镜像下载...")

# 2. 下载数据集 (VQA-RAD)
print("\n📦 正在下载 VQA-RAD 数据集...")
try:
    # 下载到当前目录下的 data_cache/vqa-rad 文件夹
    snapshot_download(
        repo_id="flaviagiammarino/vqa-rad", 
        repo_type="dataset",
        local_dir="./data_cache/vqa-rad",
        resume_download=True # 支持断点续传
    )
    print("✅ 数据集下载成功！")
except Exception as e:
    print(f"❌ 数据集下载失败: {e}")

# 3. 下载模型 (ViLT)
print("\n🤖 正在下载 ViLT 模型...")
try:
    # 下载到当前目录下的 model_cache/vilt 文件夹
    snapshot_download(
        repo_id="dandelin/vilt-b32-mlm", 
        local_dir="./model_cache/vilt",
        resume_download=True
    )
    print("✅ 模型下载成功！")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")

print("\n🎉 所有资源准备就绪！")