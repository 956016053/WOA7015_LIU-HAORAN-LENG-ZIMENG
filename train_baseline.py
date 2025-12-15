import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter

# 引入我们之前写的两个文件
from data_loader import VQARADDataset
from model_baseline import MedVQABaseline


# --- 1. 简单的工具函数：构建词表 ---
def build_vocab(dataset, min_freq=1):
    """
    遍历训练集所有问题，建立单词到数字的映射 (Word -> Index)
    """
    print("正在构建词表...")
    counter = Counter()
    for item in dataset:
        tokens = item['question_text'].lower().replace('?', '').split()
        counter.update(tokens)

    # 0留给padding (填充), 1留给unknown (未知词)
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"词表构建完成，共 {len(vocab)} 个词")
    return vocab


# --- 2. 文本处理函数 ---
def text_pipeline(text, vocab, max_len=20):
    """将文本句子转换为数字索引列表"""
    tokens = text.lower().replace('?', '').split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]

    # 截断或填充到固定长度
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices, dtype=torch.long)


# --- 3. 自定义 collate_fn ---
# DataLoader取数据时会调用这个函数来打包一个 Batch
def my_collate_fn(batch):
    # batch 是一个列表，里面是 dataset.__getitem__ 返回的字典
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    raw_questions = [item['question_text'] for item in batch]

    # 在这里把文本转成 Tensor
    # 注意：这里的 vocab 是全局变量，实际工程中最好传参，为了简单我们先这样写
    question_tensors = torch.stack([text_pipeline(q, vocab) for q in raw_questions])

    return images, question_tensors, labels


# --- 主训练程序 ---
if __name__ == "__main__":
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    BATCH_SIZE = 16  # 如果显存不够，改小一点，比如 8
    LEARNING_RATE = 1e-4
    EPOCHS = 10  # 训练轮数，先试跑10轮

    # 1. 准备数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = VQARADDataset(split='train', transform=transform)

    # 【关键修改】加载测试集时，传入训练集的 answer_map
    test_ds = VQARADDataset(split='test', transform=transform, answer_map=train_ds.ans2label)

    # 构建词表 (只用训练集构建)
    vocab = build_vocab(train_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate_fn)

    # 2. 初始化模型
    model = MedVQABaseline(vocab_size=len(vocab), num_classes=train_ds.num_classes)
    model = model.to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS} 开始...")

        for i, (imgs, q_indices, labels) in enumerate(train_loader):
            imgs, q_indices, labels = imgs.to(device), q_indices.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(imgs, q_indices)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 0:  # 每10个batch打印一次
                print(f"  Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        print(f"Epoch {epoch + 1} 结束. Avg Loss: {total_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # --- 验证/测试 ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, q_indices, labels in test_loader:
                imgs, q_indices, labels = imgs.to(device), q_indices.to(device), labels.to(device)
                outputs = model(imgs, q_indices)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%")

    # 5. 保存模型
    torch.save(model.state_dict(), "baseline_model.pth")
    print("模型已保存为 baseline_model.pth")