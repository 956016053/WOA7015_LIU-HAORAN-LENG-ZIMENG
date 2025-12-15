import torch
import torch.nn as nn
import torchvision.models as models


class MedVQABaseline(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=256, hidden_dim=512):
        """
        Args:
            vocab_size: 词表大小 (问题里有多少种不同的单词)
            num_classes: 答案类别数 (数据加载时算出来的 432)
            embed_dim: 词向量维度
            hidden_dim: 隐藏层维度
        """
        super(MedVQABaseline, self).__init__()

        # 1. 图像编码器 (Image Encoder) - 使用预训练的 ResNet18
        # 我们去掉最后全连接层，只用它来提取特征
        resnet = models.resnet18(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(512, hidden_dim)  # 把图片特征转换维度

        # 2. 文本编码器 (Text Encoder) - 使用 LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)

        # 3. 融合与分类 (Fusion & Classifier)
        # 将图片特征和文本特征拼接，然后分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, question_indices):
        """
        images: [batch_size, 3, 224, 224]
        question_indices: [batch_size, seq_len] (单词的索引)
        """
        # --- 处理图像 ---
        # ResNet 输出: [batch, 512, 1, 1] -> 展平为 [batch, 512]
        img_features = self.resnet_features(images)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.img_proj(img_features)  # [batch, hidden_dim]

        # --- 处理文本 ---
        # 词向量嵌入
        embeds = self.embedding(question_indices)  # [batch, seq_len, embed_dim]
        # LSTM 输出: output, (hidden, cell)
        # 我们只取最后时刻的 hidden state 作为整句话的特征
        _, (hidden, _) = self.lstm(embeds)
        text_features = hidden[-1]  # [batch, hidden_dim]

        # --- 融合 (拼接) ---
        combined = torch.cat((img_features, text_features), dim=1)  # [batch, hidden_dim * 2]

        # --- 分类 ---
        output = self.classifier(combined)  # [batch, num_classes]

        return output


# --- 测试代码 (运行这个文件时会执行) ---
if __name__ == "__main__":
    # 模拟输入数据
    # 假设词表有 1000 个词，Batch Size 为 4，问题长度为 10
    vocab_size = 1000
    num_classes = 432
    model = MedVQABaseline(vocab_size, num_classes)

    dummy_image = torch.randn(4, 3, 224, 224)  # 随机生成图片数据
    dummy_questions = torch.randint(0, 1000, (4, 10))  # 随机生成问题索引

    # 前向传播
    output = model(dummy_image, dummy_questions)

    print("模型构建成功！")
    print(f"输入图片形状: {dummy_image.shape}")
    print(f"输入问题形状: {dummy_questions.shape}")
    print(f"输出预测形状: {output.shape} (应为 [4, 432])")