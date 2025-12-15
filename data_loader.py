import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import os


class VQARADDataset(Dataset):
    def __init__(self, split='train', transform=None, tokenizer=None, max_length=32, answer_map=None):
        """
        新增参数: answer_map (dict): 强制使用指定的答案映射表
        """
        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # --- 修复部分开始 ---
        if answer_map is None:
            # 如果没有提供映射表（通常是训练集），则根据当前数据建立一个新的
            all_answers = set(self.dataset['answer'])
            self.ans2label = {ans: i for i, ans in enumerate(sorted(all_answers))}
        else:
            # 如果提供了映射表（通常是测试集），直接使用训练集的映射
            self.ans2label = answer_map

        self.label2ans = {i: ans for ans, i in self.ans2label.items()}
        # 注意：类别总数必须与映射表一致
        self.num_classes = len(self.ans2label)
        # --- 修复部分结束 ---

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 1. 处理图像
        image = item['image']
        # 确保图像是 RGB 模式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 2. 处理文本 (问题)
        question = item['question']
        # 如果提供了 tokenizer (后续 VLM 阶段会用到)，这里就进行 tokenization
        # 现阶段我们先返回原始文本，稍后在模型内部处理

        # 3. 处理标签 (答案)
        answer_text = item['answer']
        label = self.ans2label.get(answer_text, -1)  # 获取类别索引

        return {
            'image': image,
            'question_text': question,
            'label': torch.tensor(label, dtype=torch.long)
        }


# --- 测试代码 ---
if __name__ == "__main__":
    # 定义基本的图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为标准大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 实例化数据集
    print("正在下载并加载数据...")
    train_ds = VQARADDataset(split='train', transform=transform)
    test_ds = VQARADDataset(split='test', transform=transform)

    print(f"训练集大小: {len(train_ds)}")
    print(f"测试集大小: {len(test_ds)}")
    print(f"答案类别总数: {train_ds.num_classes}")

    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    # 查看一个 batch 的数据
    batch = next(iter(train_loader))
    print("\n--- Batch Info ---")
    print(f"Image shape: {batch['image'].shape}")  # 应为 [4, 3, 224, 224]
    print(f"Questions: {batch['question_text']}")
    print(f"Labels: {batch['label']}")