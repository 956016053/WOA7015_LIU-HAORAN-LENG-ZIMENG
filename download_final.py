import os
# 1. 强制走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("🚀 正在下载资源 (走国内镜像)...")

# 下载数据集
try:
    print("📦 Downloading VQA-RAD...")
    snapshot_download(repo_id="flaviagiammarino/vqa-rad", repo_type="dataset", local_dir="./data_cache/vqa-rad", resume_download=True)
except Exception as e:
    print(f"Dataset Error: {e}")

# 下载模型
try:
    print("🤖 Downloading ViLT...")
    snapshot_download(repo_id="dandelin/vilt-b32-mlm", local_dir="./model_cache/vilt", resume_download=True)
except Exception as e:
    print(f"Model Error: {e}")

print("✅ 下载完成！")
