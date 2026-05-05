"""
YOLOv11-ViT Hybrid Model for Pedestrian Detection in Indian Road Conditions.

Architecture Rationale:
- YOLOv11 Backbone: Fast, accurate local feature extraction
- Vision Transformer Neck: Global context modeling to handle heavy occlusions
  (e.g., pedestrians behind auto-rickshaws, dense crowd scenarios)
- Fusion: Combines dense YOLO features with sparse ViT attention maps

Why ViT in the Neck?
1. Pedestrians in India are often occluded by vehicles/traffic
2. ViT's self-attention models global relationships (sees the whole scene)
3. YOLO alone struggles with partial visibility; ViT provides context clues
4. Minimal latency overhead compared to multi-scale YOLO approaches
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from dataclasses import dataclass
import model_config

@dataclass
class ModelOutput:
    predictions: torch.Tensor
    features: Dict[str, torch.Tensor]
    vit_attention: Optional[torch.Tensor] = None

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, (H, W)

class VisionTransformerNeck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, in_channels)
        self.attention_maps = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patches, (grid_h, grid_w) = self.patch_embed(x)
        patches = patches + self.pos_embed
        patches = self.pos_drop(patches)
        attn_maps = []
        for block in self.transformer_blocks:
            patches, attn = block(patches)
            attn_maps.append(attn)
        self.attention_maps = torch.stack(attn_maps, dim=0)
        patches = self.norm(patches)
        patches = self.proj_out(patches)
        out = rearrange(patches, 'b (h w) c -> b c h w', h=grid_h, w=grid_w)
        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.attention_weights = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        self.attention_weights = attn_weights
        return x, attn_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, 'b n t h d -> t b h n d')
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn

class YOLOv11ViTHybrid(nn.Module):
    def __init__(self, config: model_config.ModelConfig) -> None:
        super().__init__()
        self.config = config
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        yolo_model_name = config.backbone.model_name.replace("yolov11", "yolov11")
        self.yolo_base = YOLO(f"{yolo_model_name}.pt")
        self.backbone = self.yolo_base.model[:10]
        self.yolo_neck = self.yolo_base.model[10:13]
        if config.use_vit_neck:
            feature_channels = 256
            self.vit_neck = VisionTransformerNeck(
                in_channels=feature_channels,
                embed_dim=config.vit_neck.embed_dim,
                num_heads=config.vit_neck.num_heads,
                depth=config.vit_neck.depth,
                patch_size=config.vit_neck.patch_size,
                mlp_ratio=config.vit_neck.mlp_ratio,
                dropout=config.vit_neck.dropout,
                attention_dropout=config.vit_neck.attention_dropout,
            )
            self.fusion_method = config.fusion_method
            if config.fusion_method == "concat":
                self.fusion_proj = nn.Conv2d(feature_channels * 2, feature_channels, 1)
            elif config.fusion_method == "add":
                pass
            elif config.fusion_method == "cross_attention":
                self.cross_attn = CrossAttentionFusion(feature_channels)
        self.yolo_head = self.yolo_base.model[13:]
        self.num_classes = config.dataset.num_classes
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        backbone_features = self.backbone(x)
        if self.config.use_vit_neck:
            yolo_neck_out = self.yolo_neck(backbone_features)
            vit_enhanced = self.vit_neck(yolo_neck_out[-1])
            if self.fusion_method == "concat":
                fused = torch.cat([yolo_neck_out[-1], vit_enhanced], dim=1)
                fused = self.fusion_proj(fused)
            elif self.fusion_method == "add":
                fused = yolo_neck_out[-1] + vit_enhanced
            elif self.fusion_method == "cross_attention":
                fused = self.cross_attn(yolo_neck_out[-1], vit_enhanced)
            yolo_neck_out = list(yolo_neck_out)
            yolo_neck_out[-1] = fused
            predictions = self.yolo_head(yolo_neck_out)
        else:
            yolo_neck_out = self.yolo_neck(backbone_features)
            predictions = self.yolo_head(yolo_neck_out)
        return ModelOutput(
            predictions=predictions,
            features={
                "backbone": backbone_features,
                "yolo_neck": yolo_neck_out,
                "vit_attention": self.vit_neck.attention_maps if hasattr(self, 'vit_neck') else None,
            },
            vit_attention=self.vit_neck.attention_maps if hasattr(self, 'vit_neck') else None,
        )

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query_proj = nn.Conv2d(channels, channels // 8, 1)
        self.key_proj = nn.Conv2d(channels, channels // 8, 1)
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(
        self,
        yolo_features: torch.Tensor,
        vit_features: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = yolo_features.shape
        q = self.query_proj(yolo_features).view(B, -1, H * W)
        k = self.key_proj(vit_features).view(B, -1, H * W)
        v = self.value_proj(vit_features).view(B, C, H * W)
        attn = torch.bmm(q.transpose(1, 2), k) / (C // 8) ** 0.5
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        out = self.out_proj(out)
        return yolo_features + out

def build_model(config: model_config.ModelConfig, device: str = "cuda") -> YOLOv11ViTHybrid:
    model = YOLOv11ViTHybrid(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: YOLOv11-ViT Hybrid")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Device: {device}")
    return model

if __name__ == "__main__":
    config = model_config.ModelConfig.from_preset("small")
    model = build_model(config, device="cpu")
    x = torch.randn(2, 3, 640, 640)
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Predictions shape: {output.predictions.shape}")
    print(f"Features keys: {output.features.keys()}")
