from ParallelCNNHead import ParallelCNNHead
import torch.nn as nn

class FakeBERT(nn.Module):
    """Encoder-agnostic — swap BERT for DeBERTa by passing a different encoder."""
    def __init__(self, encoder, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.encoder    = encoder
        self.cnn_head   = ParallelCNNHead(hidden_size)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        out      = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out  = out.last_hidden_state          # (B, seq_len, hidden)

        # Force seq_out to match the CNN head's parameters
        seq_out = seq_out.to(self.classifier.weight.dtype)

        features = self.cnn_head(seq_out)         # (B, 128)
        logits   = self.classifier(features)      # (B, 2)
        return logits