import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from collections import Counter
import re
import os

class SimpleTokenizer:
    def __init__(self, texts, max_vocab=5000):
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for word, _ in word_counts.most_common(max_vocab - 2):
            self.vocab[word] = len(self.vocab)
        self.idx2word = {i: w for w, i in self.vocab.items()}
    def _tokenize(self, text):
        return re.findall(r"\w+", str(text).lower())
    def encode(self, text, max_len=32):
        words = self._tokenize(text)
        indices = [self.vocab.get(w, 1) for w in words[:max_len]]
        if len(indices) < max_len: indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    def __len__(self): return len(self.vocab)

class VQARADDataset(Dataset):
    def __init__(self, split, answer_map=None, tokenizer=None, processor=None, transform=None):
        data_path = "./data_cache/vqa-rad"
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"找不到数据集，请先运行 download_final.py")
             
        self.dataset = load_dataset(data_path, split=split)
        self.transform = transform
        self.processor = processor
        self.tokenizer = tokenizer
        
        if answer_map is None: self.answer_map = self._build_answer_map()
        else: self.answer_map = answer_map
        self.id2answer = {v: k for k, v in self.answer_map.items()}

    def _build_answer_map(self):
        answers = set([str(item['answer']).lower() for item in self.dataset])
        return {ans: i for i, ans in enumerate(sorted(list(answers)))}
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        question = str(item['question']).lower()
        answer = str(item['answer']).lower()
        if image.mode != 'RGB': image = image.convert('RGB')
        label = self.answer_map.get(answer, 0)

        # === 【关键修复】 ===
        # ViLT 默认处理器会保留长宽比，导致 Batch 堆叠失败
        # 这里强制把图片 Resize 成 384x384 (ViLT 的标准输入尺寸)
        if self.processor:
            image = image.resize((384, 384)) # 强制拉伸
            encoding = self.processor(image, question, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
            for k,v in encoding.items(): encoding[k] = v.squeeze()
            encoding['labels'] = torch.tensor(label)
            return encoding
        
        if self.transform and self.tokenizer:
            img_tensor = self.transform(image)
            text_tensor = self.tokenizer.encode(question)
            return {"image": img_tensor, "text": text_tensor, "label": torch.tensor(label)}

baseline_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
