import torch
import torch.nn as nn
from torchvision import models

class BaselineCNNLSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=300, hidden_dim=512):
        super(BaselineCNNLSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, 512)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, images, text_indices):
        v_feat = self.visual_fc(self.resnet_features(images).view(images.size(0), -1))
        _, (hidden, _) = self.lstm(self.embedding(text_indices))
        t_feat = hidden[-1]
        return self.classifier(torch.cat((v_feat, t_feat), dim=1))
