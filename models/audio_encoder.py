import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AudioEncoderV2(nn.Module):
    """增强版音频编码器，支持多语言音频特征提取"""
    def __init__(self, input_dim=40, hidden_dim=512, output_dim=512, num_layers=3, num_heads=4):
        super().__init__()
        self.input_dim = input_dim  # MFCC特征维度
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim))  # 最大序列长度1000
        
        # 卷积层（捕捉局部特征）
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            ) for _ in range(2)
        ])
        self.conv_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(2)
        ])
        
        # Transformer编码器层（捕捉全局特征）
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 语言自适应层（多语言支持）
        self.lang_adapt = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),  # 10维语言嵌入
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.lang_embedding = nn.Embedding(10, 10)  # 支持10种语言
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, lang_id=None):
        """
        参数:
            x: 音频特征（如MFCC），形状为 (batch_size, seq_len, input_dim)
            lang_id: 语言ID，形状为 (batch_size,) 或 None
            
        返回:
            提取的音频特征，形状为 (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 输入投影
        x = self.input_proj(x)  # (B, T, H)
        
        # 2. 添加位置编码
        if seq_len > self.pos_encoder.size(1):
            # 如果序列长度超过预定义的最大长度，使用线性插值扩展位置编码
            pos_encoder = F.interpolate(
                self.pos_encoder.permute(0, 2, 1),
                size=seq_len,
                mode='linear'
            ).permute(0, 2, 1)
        else:
            pos_encoder = self.pos_encoder[:, :seq_len]
        
        x = x + pos_encoder  # (B, T, H)
        
        # 3. 卷积层处理
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            x_conv = conv(x.permute(0, 2, 1))  # (B, H, T)
            x_conv = norm(x_conv)
            x_conv = F.gelu(x_conv)
            x = x + x_conv.permute(0, 2, 1)  # (B, T, H) 残差连接
        
        # 4. Transformer处理
        for layer in self.transformer_layers:
            x = layer(x)  # (B, T, H)
        
        # 5. 语言自适应（如果提供了语言ID）
        if lang_id is not None:
            # 扩展语言嵌入到序列长度
            lang_emb = self.lang_embedding(lang_id)  # (B, 10)
            lang_emb = lang_emb.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, 10)
            
            # 融合语言信息
            x = self.lang_adapt(torch.cat([x, lang_emb], dim=-1))  # (B, T, H)
        
        # 6. 输出投影
        output = self.output_proj(x)  # (B, T, O)
        
        return output
