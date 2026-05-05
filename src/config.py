"""
Pydantic-based Configuration Management
Handles model, training, and data configurations with type safety and validation.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


# ============================================================================
# ENUMS
# ============================================================================

class DatasetEnum(str, Enum):
    """Supported datasets."""
    UVH26 = "uvh26"
    COCO = "coco"
    CUSTOM = "custom"


class BackboneEnum(str, Enum):
    """Supported backbone architectures."""
    YOLO11N = "yolo11n"
    YOLO11S = "yolo11s"
    YOLO11M = "yolo11m"


class TaskEnum(str, Enum):
    """Detection task type."""
    DETECT = "detect"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ViTConfig(BaseModel):
    """Vision Transformer block configuration."""
    
    hidden_dim: int = Field(512, description="Transformer hidden dimension")
    num_heads: int = Field(8, description="Number of attention heads")
    num_layers: int = Field(3, description="Number of transformer layers")
    mlp_ratio: float = Field(4.0, description="MLP hidden dimension ratio")
    dropout: float = Field(0.1, description="Dropout rate")
    attention_dropout: float = Field(0.1, description="Attention dropout rate")
    patch_size: int = Field(16, description="Patch size for ViT")
    
    class Config:
        frozen = True


class YOLONeckConfig(BaseModel):
    """YOLO neck (feature pyramid) configuration."""
    
    depth_multiple: float = Field(1.0, description="Depth multiplier for neck")
    width_multiple: float = Field(1.0, description="Width multiplier for neck")
    vit_insertion_layer: int = Field(1, description="Layer to insert ViT block (0-indexed)")
    
    class Config:
        frozen = True


class BackboneConfig(BaseModel):
    """YOLO11 backbone configuration."""
    
    name: BackboneEnum = Field(BackboneEnum.YOLO11S, description="Backbone model size")
    pretrained: bool = Field(True, description="Use pretrained weights")
    freeze_backbone: bool = Field(False, description="Freeze backbone during training")
    
    class Config:
        frozen = True


class HeadConfig(BaseModel):
    """Detection head configuration."""
    
    num_classes: int = Field(1, description="Number of classes (1 for pedestrian)")
    stride: List[int] = Field([8, 16, 32], description="Detection strides")
    
    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Complete model configuration."""
    
    task: TaskEnum = Field(TaskEnum.DETECT, description="Detection task")
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    neck: YOLONeckConfig = Field(default_factory=YOLONeckConfig)
    vit: ViTConfig = Field(default_factory=ViTConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)
    
    # Input specs
    input_size: int = Field(640, description="Input image size (square)")
    input_channels: int = Field(3, description="Input channels (RGB)")
    
    # Model behavior
    use_mixed_precision: bool = Field(True, description="Use FP16 for faster inference")
    device: str = Field("cuda", description="Device: cuda or cpu")
    
    class Config:
        frozen = True
    
    @property
    def vit_enabled(self) -> bool:
        """Check if ViT block is enabled."""
        return True  # Always enabled in this hybrid model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class AugmentationConfig(BaseModel):
    """Data augmentation configuration for Indian road conditions."""
    
    # Standard augmentations
    hflip: float = Field(0.5, description="Horizontal flip probability")
    vflip: float = Field(0.0, description="Vertical flip probability")
    rotate: Tuple[int, int] = Field((-15, 15), description="Rotation angle range")
    
    # India-specific augmentations
    dust_haze: float = Field(0.4, description="Dust/haze probability (summer conditions)")
    motion_blur: float = Field(0.3, description="Motion blur probability (traffic simulation)")
    rain: float = Field(0.2, description="Rain probability (monsoon simulation)")
    brightness: Tuple[float, float] = Field((0.7, 1.3), description="Brightness range")
    
    # Geometric
    scale: Tuple[float, float] = Field((0.8, 1.2), description="Scale range")
    translate: Tuple[float, float] = Field((0.1, 0.1), description="Translation range")
    
    class Config:
        frozen = True


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    
    name: DatasetEnum = Field(DatasetEnum.UVH26, description="Dataset name")
    data_root: Path = Field(Path("data/raw/uvh26"), description="Dataset root path")
    
    # UVH-26 specific classes
    classes: List[str] = Field(
        ["pedestrian"],
        description="Target classes (UVH-26 uses pedestrian as primary class)"
    )
    num_classes: int = Field(1, description="Number of classes")
    
    # Data split
    train_val_split: float = Field(0.8, description="Train/val split ratio")
    test_size: float = Field(0.1, description="Test set size")
    
    # Loading
    batch_size: int = Field(32, description="Batch size")
    num_workers: int = Field(4, description="Number of data loading workers")
    pin_memory: bool = Field(True, description="Pin memory for faster loading")
    
    # Augmentation
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True
    
    @validator("data_root", pre=True, always=True)
    def validate_path(cls, v):
        """Ensure path is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    
    name: str = Field("adamw", description="Optimizer name")
    lr: float = Field(1e-4, description="Learning rate")
    lr_backbone: float = Field(1e-5, description="Backbone learning rate (if frozen)")
    weight_decay: float = Field(1e-4, description="Weight decay (L2 regularization)")
    momentum: float = Field(0.9, description="Momentum (for SGD)")
    
    class Config:
        frozen = True


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    
    name: str = Field("cosine", description="Scheduler name")
    warmup_epochs: int = Field(5, description="Number of warmup epochs")
    warmup_lr: float = Field(1e-5, description="Initial warmup learning rate")
    
    class Config:
        frozen = True


class LossConfig(BaseModel):
    """Loss function weights."""
    
    conf_loss_weight: float = Field(1.0, description="Objectness loss weight")
    cls_loss_weight: float = Field(0.5, description="Classification loss weight")
    bbox_loss_weight: float = Field(7.5, description="Bounding box loss weight")
    vit_regularization: float = Field(0.01, description="ViT L2 regularization")
    
    class Config:
        frozen = True


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    
    # Training loop
    epochs: int = Field(100, description="Number of training epochs")
    resume_from: Optional[Path] = Field(None, description="Resume from checkpoint")
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    
    # Validation
    val_interval: int = Field(5, description="Validate every N epochs")
    save_interval: int = Field(10, description="Save checkpoint every N epochs")
    
    # Metrics
    log_interval: int = Field(100, description="Log metrics every N batches")
    tensorboard_dir: Path = Field(Path("logs/"), description="TensorBoard log directory")
    
    # Device
    device: str = Field("cuda", description="Training device")
    mixed_precision: bool = Field(True, description="Use mixed precision training")
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = Field(1.0, description="Gradient clipping norm")
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

class InferenceConfig(BaseModel):
    """Inference configuration."""
    
    conf_threshold: float = Field(0.5, description="Confidence threshold")
    iou_threshold: float = Field(0.5, description="NMS IoU threshold")
    max_detections: int = Field(300, description="Maximum detections per image")
    
    # Performance
    use_half: bool = Field(True, description="Use FP16 inference")
    use_tensorrt: bool = Field(False, description="Use TensorRT for inference")
    
    # Latency tracking
    track_latency: bool = Field(True, description="Log inference latency")
    warmup_runs: int = Field(5, description="Warmup runs before latency tracking")
    
    class Config:
        frozen = True


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================

class Config(BaseModel):
    """Master configuration combining all sub-configs."""
    
    # Project metadata
    project_name: str = Field("pedestrian-detection-india")
    version: str = Field("1.0.0")
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    
    # Paths
    root_dir: Path = Field(Path("."))
    weights_dir: Path = Field(Path("weights/"))
    
    class Config:
        arbitrary_types_allowed = True
        frozen = True
    
    @root_validator(pre=False)
    def validate_config(cls, values):
        """Cross-validate configuration."""
        model_cfg = values.get("model")
        dataset_cfg = values.get("dataset")
        
        # Ensure model and dataset num_classes match
        if model_cfg and dataset_cfg:
            if model_cfg.head.num_classes != dataset_cfg.num_classes:
                raise ValueError(
                    f"Model num_classes ({model_cfg.head.num_classes}) must match "
                    f"dataset num_classes ({dataset_cfg.num_classes})"
                )
        
        return values
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(output_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_default_config() -> Config:
    """Get default configuration for pedestrian detection."""
    return Config()


def get_vit_config_for_backbone(backbone_size: BackboneEnum) -> ViTConfig:
    """Get optimized ViT configuration based on backbone size."""
    configs = {
        BackboneEnum.YOLO11N: ViTConfig(hidden_dim=256, num_heads=4, num_layers=2),
        BackboneEnum.YOLO11S: ViTConfig(hidden_dim=512, num_heads=8, num_layers=3),
        BackboneEnum.YOLO11M: ViTConfig(hidden_dim=768, num_heads=12, num_layers=4),
    }
    return configs.get(backbone_size, ViTConfig())
