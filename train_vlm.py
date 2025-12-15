import torch
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.optim import AdamW  # 使用 PyTorch 原生优化器
from tqdm import tqdm
import torch.nn as nn
from data_loader import VQARADDataset

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 5e-5
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


# --- 2. 数据处理 ---
def vlm_collate_fn(batch):
    images = [item['image'] for item in batch]
    questions = [item['question_text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    encoding = processor(images, questions, return_tensors="pt", padding=True, truncation=True)
    return encoding, labels


def main():
    print(f"使用设备: {device}")

    # --- 3. 准备数据 ---
    train_ds = VQARADDataset(split='train', transform=None)
    test_ds = VQARADDataset(split='test', transform=None, answer_map=train_ds.ans2label)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=vlm_collate_fn)

    num_classes = train_ds.num_classes
    print(f"分类类别数: {num_classes}")

    # --- 4. 加载模型 ---
    print("正在加载 ViLT 预训练模型...")
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-mlm",
        num_labels=num_classes,
        id2label=train_ds.label2ans,
        label2id=train_ds.ans2label,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # --- 5. 优化器与损失函数 ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()  # 定义单分类损失函数

    # --- 6. 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for batch_encoding, labels in loop:
            input_ids = batch_encoding['input_ids'].to(device)
            pixel_values = batch_encoding['pixel_values'].to(device)
            token_type_ids = batch_encoding['token_type_ids'].to(device)
            attention_mask = batch_encoding['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播 (不传 labels，只拿 logits)
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

            logits = outputs.logits
            loss = criterion(logits, labels)  # 计算 Loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Avg Loss: {total_loss / len(train_loader):.4f}")

        # --- 7. 验证 ---
        model.eval()
        correct = 0
        total = 0
        print("正在验证...")
        with torch.no_grad():
            for batch_encoding, labels in test_loader:
                input_ids = batch_encoding['input_ids'].to(device)
                pixel_values = batch_encoding['pixel_values'].to(device)
                token_type_ids = batch_encoding['token_type_ids'].to(device)
                attention_mask = batch_encoding['attention_mask'].to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

                logits = outputs.logits
                _, predicted = torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "vlm_model.pth")
    print("VLM 模型已保存！")


if __name__ == "__main__":
    main()
