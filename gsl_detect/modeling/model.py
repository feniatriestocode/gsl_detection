import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) 

        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class GSLTransformer(nn.Module):
    def __init__ (
        self,
        input_dim=390,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.5,
        num_classes=None
    ):
        super().__init__()
        # Projection and positional encoding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        #Transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True # Pre-LN
        )

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) 
        )



    def forward(self, x, mask):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        src_key_padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        mask_expanded = mask.unsqueeze(-1).float() 
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        logits = self.classifier(pooled)
        return logits
    