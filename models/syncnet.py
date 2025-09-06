import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SyncNetV2(nn.Module):
    """增强版同步检测网络，判断唇形与音频是否同步"""
    def __init__(self, audio_dim=512, video_dim=512, hidden_dim=256, num_heads=4):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        
        # 音频特征处理
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 视频特征处理
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 自注意力（音频）
        self.audio_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 自注意力（视频）
        self.video_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 时间卷积（捕捉局部时间依赖）
        self.time_conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        # 时间偏移预测器（预测音频和视频之间的时间偏移）
        self.offset_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 11)  # 预测-5到+5帧的偏移
        )
        
        # 同步分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)  # 0: 不同步, 1: 同步
        )
        
        # 相似度计算层
        self.similarity_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, audio_features, video_features):
        """
        参数:
            audio_features: 音频特征，形状为 (batch_size, seq_len, audio_dim)
            video_features: 视频特征，形状为 (batch_size, seq_len, video_dim)
            
        返回:
            包含分类结果、相似度分数和时间偏移预测的字典
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # 特征投影
        audio_proj = self.audio_proj(audio_features)  # (B, T, H)
        video_proj = self.video_proj(video_features)  # (B, T, H)
        
        # 自注意力
        audio_self_attn, _ = self.audio_self_attn(audio_proj, audio_proj, audio_proj)
        video_self_attn, _ = self.video_self_attn(video_proj, video_proj, video_proj)
        
        # 跨模态注意力（音频关注视频）
        audio_cross_attn, _ = self.cross_attention(
            audio_self_attn, video_self_attn, video_self_attn
        )
        
        # 跨模态注意力（视频关注音频）
        video_cross_attn, _ = self.cross_attention(
            video_self_attn, audio_self_attn, audio_self_attn
        )
        
        # 融合跨模态特征
        cross_feat = torch.cat([audio_cross_attn, video_cross_attn], dim=-1)  # (B, T, 2H)
        
        # 时间卷积
        cross_feat_conv = self.time_conv(
            cross_feat.permute(0, 2, 1)  # 转为 (B, 2H, T)
        ).permute(0, 2, 1)  # 转回 (B, T, H)
        
        # 计算帧级别相似度
        frame_similarity = self.similarity_layer(
            F.gelu(cross_feat_conv)
        ).squeeze(-1)  # (B, T)
        
        # 平均时间维度得到序列级特征
        seq_feat = torch.mean(cross_feat_conv, dim=1)  # (B, H)
        
        # 预测时间偏移
        offset_logits = self.offset_predictor(seq_feat)  # (B, 11)
        
        # 同步分类
        logits = self.classifier(seq_feat)  # (B, 2)
        
        # 计算同步分数（平均相似度）
        sync_score = torch.mean(frame_similarity, dim=1)  # (B,)
        
        return {
            "logits": logits,  # 分类logits
            "sync_score": sync_score,  # 同步分数
            "frame_similarity": frame_similarity,  # 帧级别相似度
            "offset_logits": offset_logits  # 时间偏移预测
        }
    
    def compute_sync_loss(self, audio_features, video_features, positive=True):
        """计算同步损失"""
        output = self.forward(audio_features, video_features)
        
        # 分类损失
        labels = torch.ones(audio_features.size(0), device=audio_features.device).long() if positive else \
                 torch.zeros(audio_features.size(0), device=audio_features.device).long()
        cls_loss = F.cross_entropy(output["logits"], labels)
        
        # 时间偏移损失（对于正样本）
        offset_loss = 0.0
        if positive:
            # 理想偏移是0（中心位置）
            offset_labels = torch.full(
                (audio_features.size(0),), 5,  # 索引5对应偏移0
                device=audio_features.device, 
                dtype=torch.long
            )
            offset_loss = F.cross_entropy(output["offset_logits"], offset_labels)
        
        return {
            "total_loss": cls_loss + 0.3 * offset_loss,
            "cls_loss": cls_loss,
            "offset_loss": offset_loss,
            "sync_score": torch.mean(output["sync_score"])
        }
