import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import matplotlib
matplotlib.use('Agg') # 强制不显示窗口，防止服务器卡死
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets
import os
from PIL import Image
import re
from collections import Counter
from torchvision import models, transforms
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- 1. 核心配置 (根据你的截图修改) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data_cache/vqa-rad"
IMG_SIZE = (384, 384)

# 你的高分模型文件名
BASELINE_PATH = "baseline_acc0.5857.pth"
VILT_PATH = "vilt_acc0.5301.pth"

print(f"🚀 启动完美评估脚本 | 设备: {DEVICE}")
print(f"📄 目标模型: \n  1. {BASELINE_PATH}\n  2. {VILT_PATH}")

# --- 2. 辅助组件 ---
class SimpleTokenizer:
    def __init__(self, texts, max_vocab=8000):
        word_counts = Counter()
        for text in texts:
            words = re.findall(r"\w+", str(text).lower())
            word_counts.update(words)
        self.vocab = {"<pad>": 0, "<unk>": 1}
        # 必须保证顺序一致，否则ID会乱
        for word, _ in word_counts.most_common(max_vocab - 2):
            self.vocab[word] = len(self.vocab)
    def encode(self, text, max_len=40):
        words = re.findall(r"\w+", str(text).lower())
        indices = [self.vocab.get(w, 1) for w in words[:max_len]]
        if len(indices) < max_len: indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    def __len__(self): return len(self.vocab)

baseline_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GodBaseline(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=300, hidden_dim=512):
        super(GodBaseline, self).__init__()
        resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1]) 
        self.visual_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3)
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + hidden_dim*2, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
    def forward(self, images, text_indices):
        v = self.visual_fc(self.resnet_features(images).view(images.size(0), -1))
        _, (hidden, _) = self.lstm(self.embedding(text_indices))
        t = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) 
        return self.classifier(torch.cat((v, t), dim=1))

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, answer_map, tokenizer=None, processor=None):
        self.dataset = dataset
        self.answer_map = answer_map
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB').resize(IMG_SIZE, resample=Image.BICUBIC)
        question = str(item['question']).lower()
        answer = str(item['answer']).lower()
        label = self.answer_map.get(answer, 0)
        
        # --- 智能推断问题类型 ---
        q_type = "OPEN"
        raw_type = str(item.get('answer_type', '')).upper()
        if not raw_type: raw_type = str(item.get('phrase_type', '')).upper()
        
        if any(x in raw_type for x in ['YES', 'NO', 'CLOSED']): q_type = "CLOSED"
        elif any(x in raw_type for x in ['OPEN', 'OTHER']): q_type = "OPEN"
        elif answer in ['yes', 'no', 'true', 'false']: q_type = "CLOSED"

        if self.processor:
            encoding = self.processor(image, question, padding="max_length", truncation=True, max_length=40, return_tensors="pt")
            for k,v in encoding.items(): encoding[k] = v.squeeze()
            return encoding, label, q_type
        
        if self.tokenizer:
            img_tensor = baseline_normalize(image)
            text_tensor = self.tokenizer.encode(question)
            return img_tensor, text_tensor, label, q_type

# --- 3. 数据还原 (关键步骤) ---
print("🔄 正在还原训练环境 (确保 Tokenizer 一致)...")
if not os.path.exists(DATA_PATH): raise FileNotFoundError("无数据！")
ds_train = load_dataset(DATA_PATH, split="train")
ds_test = load_dataset(DATA_PATH, split="test")
full_ds = concatenate_datasets([ds_train, ds_test])

all_answers = set([str(x['answer']).lower() for x in full_ds])
answer_map = {ans: i for i, ans in enumerate(sorted(list(all_answers)))}
idx2answer = {i: ans for ans, i in answer_map.items()}
num_classes = len(answer_map)

# 切分 (Seed必须是42)
split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
train_set = split_ds['train']
test_set = split_ds['test']
print(f"✅ 测试集数量: {len(test_set)}")

# 构建 Tokenizer (必须用 Train+Test 的全部文本，顺序不能变)
all_text = [str(x['question']) for x in train_set] + [str(x['question']) for x in test_set]
tokenizer = SimpleTokenizer(all_text)

# --- 4. 评估逻辑 ---
def evaluate_model(model_name, model, dataloader):
    print(f"\n📊 评估 {model_name}...")
    model.eval()
    preds_list, labels_list, types_list = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if model_name == "ViLT":
                input_ids = batch[0]['input_ids'].to(DEVICE)
                pixel_values = batch[0]['pixel_values'].to(DEVICE)
                labels = batch[1].numpy()
                q_types = batch[2]
                outputs = model(input_ids=input_ids, pixel_values=pixel_values)
                logits = outputs.logits
            else:
                images = batch[0].to(DEVICE)
                text = batch[1].to(DEVICE)
                labels = batch[2].numpy()
                q_types = batch[3]
                logits = model(images, text)
            
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.extend(p)
            labels_list.extend(labels)
            types_list.extend(q_types)
            
    return np.array(preds_list), np.array(labels_list), np.array(types_list)

results = {}

# 加载 Baseline
if os.path.exists(BASELINE_PATH):
    try:
        baseline_ds = EvalDataset(test_set, answer_map, tokenizer=tokenizer)
        baseline_loader = DataLoader(baseline_ds, batch_size=32, shuffle=False, num_workers=4)
        model = GodBaseline(num_classes, len(tokenizer))
        model.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        model.to(DEVICE)
        results['Baseline'] = evaluate_model("Baseline", model, baseline_loader)
        del model
    except Exception as e: print(f"❌ Baseline 载入失败: {e}")

# 加载 ViLT
if os.path.exists(VILT_PATH):
    try:
        processor = ViltProcessor.from_pretrained("./model_cache/vilt")
        vilt_ds = EvalDataset(test_set, answer_map, processor=processor)
        vilt_loader = DataLoader(vilt_ds, batch_size=32, shuffle=False, num_workers=4)
        config = ViltConfig.from_pretrained("./model_cache/vilt")
        config.num_labels = num_classes
        config.id2label = idx2answer
        config.label2id = answer_map
        model = ViltForQuestionAnswering.from_pretrained("./model_cache/vilt", config=config, ignore_mismatched_sizes=True)
        model.load_state_dict(torch.load(VILT_PATH, map_location=DEVICE))
        model.to(DEVICE)
        results['ViLT'] = evaluate_model("ViLT", model, vilt_loader)
        del model
    except Exception as e: print(f"❌ ViLT 载入失败: {e}")

# --- 5. 统计与绘图 (防报错版) ---
if not results: exit()

metrics = []
for name, (p, l, t) in results.items():
    acc = accuracy_score(l, p)
    prec, rec, f1, _ = precision_recall_fscore_support(l, p, average='weighted', zero_division=0)
    
    t = np.array(t)
    mask_closed = (t == 'CLOSED')
    mask_open = (t == 'OPEN')
    acc_closed = accuracy_score(l[mask_closed], p[mask_closed]) if mask_closed.any() else 0
    acc_open = accuracy_score(l[mask_open], p[mask_open]) if mask_open.any() else 0
    
    metrics.append({
        'Model': name, 'Accuracy': acc, 'F1-Score': f1, 
        'Precision': prec, 'Recall': rec,
        'Acc (Closed)': acc_closed, 'Acc (Open)': acc_open
    })

df = pd.DataFrame(metrics)
print("\n📝 最终结果:")
print(df)
df.to_csv("evaluation_results.csv", index=False)

sns.set_style("whitegrid")

# 图1: 整体指标
try:
    print("🎨 生成 metrics_comparison.png ...")
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(id_vars="Model", value_vars=['Accuracy', 'F1-Score', 'Precision', 'Recall'], var_name="Metric", value_name="Score")
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.ylim(0, 1.0); [ax.bar_label(i, fmt='%.2f') for i in ax.containers]
    plt.title("Overall Performance Comparison")
    plt.savefig("metrics_comparison.png", dpi=300); plt.close()
    print("✅ 成功")
except Exception as e: print(f"❌ 失败: {e}")

# 图2: 题型准确率 (手动构建数据，防止 melt 报错)
try:
    print("🎨 生成 type_accuracy.png ...")
    plt.figure(figsize=(8, 6))
    # 手动构建简单的 DataFrame 用于绘图
    plot_data = []
    for i, row in df.iterrows():
        plot_data.append({"Model": row["Model"], "Type": "Closed", "Score": row["Acc (Closed)"]})
        plot_data.append({"Model": row["Model"], "Type": "Open", "Score": row["Acc (Open)"]})
    
    df_plot = pd.DataFrame(plot_data)
    ax = sns.barplot(data=df_plot, x="Type", y="Score", hue="Model", palette="magma")
    plt.ylim(0, 1.0); [ax.bar_label(i, fmt='%.2f') for i in ax.containers]
    plt.title("Accuracy by Question Type")
    plt.savefig("type_accuracy.png", dpi=300); plt.close()
    print("✅ 成功")
except Exception as e: print(f"❌ 失败: {e}")

# 图3: 混淆矩阵
try:
    print("🎨 生成 confusion_matrix ...")
    for name, (p, l, t) in results.items():
        counts = Counter(l)
        top_idx = [k for k,v in counts.most_common(15)]
        if not top_idx: continue
        top_labels = [idx2answer[k] for k in top_indices] if 'idx2answer' in globals() else top_idx # Fallback
        
        # 重新获取标签文字
        top_labels_text = [idx2answer[k] for k in top_idx]
        
        mask = np.isin(l, top_idx)
        cm = confusion_matrix(l[mask], p[mask], labels=top_idx)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_labels_text, yticklabels=top_labels_text)
        plt.title(f"Confusion Matrix (Top 15) - {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(f"confusion_matrix_{name}.png", dpi=300); plt.close()
    print("✅ 成功")
except Exception as e: print(f"❌ 失败: {e}")

print("\n🎉 全部图表生成完毕！请刷新文件栏查看！")