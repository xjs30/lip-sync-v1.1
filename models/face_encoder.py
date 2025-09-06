import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FaceEncoderV2(nn.Module):
    """增强版面部特征编码器，融合图像和特征点信息"""
    def __init__(self, input_dim=136, hidden_dim=512, output_dim=512, use_attention=True):
        super().__init__()
        self.input_dim = input_dim  # 68个特征点×2坐标
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # 特征点特征提取
        self.landmark_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 唇部特征点注意力（关注唇部区域）
        self.lip_attention = nn.Sequential(
            nn.Linear(input_dim, 68),  # 68个特征点
            nn.Sigmoid()
        )
        
        # 图像特征处理（假设输入图像特征已经提取）
        self.image_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 跨模态注意力（融合特征点和图像特征）
        if self.use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
        
        # 时序特征处理
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, landmarks, image_features=None):
        """
        参数:
            landmarks: 面部特征点，形状为 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            image_features: 面部图像特征，形状为 (batch_size, seq_len, 3, H, W) 或 (batch_size, 3, H, W)
            
        返回:
            融合的面部特征，形状为 (batch_size, seq_len, output_dim) 或 (batch_size, output_dim)
        """
        # 处理单帧情况（添加时间维度）
        has_seq_dim = landmarks.dim() == 3
        if not has_seq_dim:
            landmarks = landmarks.unsqueeze(1)  # (B, 1, D)
            if image_features is not None:
                image_features = image_features.unsqueeze(1)  # (B, 1, 3, H, W)
        
        batch_size, seq_len, _ = landmarks.shape
        
        # 1. 处理特征点
        # 重塑特征点为 (B*T, D) 以便并行处理
        landmarks_flat = rearrange(landmarks, 'b t d -> (b t) d')
        
        # 计算唇部注意力权重
        lip_weights = self.lip_attention(landmarks_flat)  # (B*T, 68)
        
        # 重塑特征点为 (B*T, 68, 2)
        landmarks_reshaped = rearrange(landmarks_flat, '(b t) (p c) -> (b t) p c', b=batch_size, t=seq_len, p=68, c=2)
        
        # 应用唇部注意力
        lip_weights_expanded = lip_weights.unsqueeze(-1)  # (B*T, 68, 1)
        landmarks_attended = landmarks_reshaped * lip_weights_expanded  # (B*T, 68, 2)
        
        # 重塑回 (B*T, D)
        landmarks_attended_flat = rearrange(landmarks_attended, '(b t) p c -> (b t) (p c)', b=batch_size, t=seq_len)
        
        # 编码特征点
        landmark_feat = self.landmark_encoder(landmarks_attended_flat)  # (B*T, H)
        landmark_feat = rearrange(landmark_feat, '(b t) h -> b t h', b=batch_size, t=seq_len)  # (B, T, H)
        
        # 2. 处理图像特征（如果提供）
        image_feat = None
        if image_features is not None:
            # 重塑图像特征为 (B*T, 3, H, W)
            image_flat = rearrange(image_features, 'b t c h w -> (b t) c h w')
            
            # 提取图像特征
            img_feat_flat = self.image_proj(image_flat)  # (B*T, H)
            image_feat = rearrange(img_feat_flat, '(b t) h -> b t h', b=batch_size, t=seq_len)  # (B, T, H)
        
        # 3. 融合特征
        if image_feat is not None:
            if self.use_attention:
                # 使用跨模态注意力融合
                fused_feat, _ = self.cross_attention(landmark_feat, image_feat, image_feat)
                fused_feat = fused_feat + landmark_feat  # 残差连接
            else:
                # 简单拼接融合
                fused_feat = torch.cat([landmark_feat, image_feat], dim=-1)
                fused_feat = F.gelu(nn.Linear(fused_feat.size(-1), self.hidden_dim).to(fused_feat.device)(fused_feat))
        else:
            fused_feat = landmark_feat
        
        # 4. 时序处理
        fused_feat = self.temporal_conv(
            fused_feat.permute(0, 2, 1)  # 转为 (B, H, T)
        ).permute(0, 2, 1)  # 转回 (B, T, H)
        
        # 5. 输出投影
        output = self.output_proj(fused_feat)  # (B, T, O)
        
        # 如果输入没有时间维度，移除时间维度
        if not has_seq_dim:
            output = output.squeeze(1)  # (B, O)
        
        return output
