import torch
import torch.nn as nn
# from torch.utils.data import Dataset

class ParallelCNNHead(nn.Module):
    """
    3 parallel 1D-CNN blocks (kernels 3,4,5) → concat → 2 more conv layers.
    Matches FakeBERT Table 5 topology from the paper.
    """
    def __init__(self, hidden_size: int, num_filters: int = 128):
        super().__init__()
        self.conv3 = nn.Conv1d(hidden_size, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_size, num_filters, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(hidden_size, num_filters, kernel_size=5, padding=2)
        self.pool  = nn.AdaptiveMaxPool1d(1)

        self.conv_post = nn.Conv1d(num_filters * 3, num_filters, kernel_size=5, padding=2)

        self.dense1  = nn.Linear(num_filters, 384)
        self.dense2  = nn.Linear(384, 128)
        self.dropout = nn.Dropout(0.2)
        self.relu    = nn.ReLU()

    def forward(self, x):
        # x: (B, seq_len, hidden) → (B, hidden, seq_len)
        x = x.permute(0, 2, 1)

        b3 = self.pool(self.relu(self.conv3(x)))   # (B, 128, 1)
        b4 = self.pool(self.relu(self.conv4(x)))
        b5 = self.pool(self.relu(self.conv5(x)))

        cat = torch.cat([b3, b4, b5], dim=1)       # (B, 384, 1)
        cat = cat.expand(-1, -1, 10)               # expand seq dim for post-conv
        out = self.pool(self.relu(self.conv_post(cat)))  # (B, 128, 1)
        out = out.squeeze(-1)                       # (B, 128)

        out = self.dropout(self.relu(self.dense1(out)))
        out = self.dropout(self.relu(self.dense2(out)))
        return out