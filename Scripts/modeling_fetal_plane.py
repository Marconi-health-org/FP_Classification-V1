# modeling_fetal_plane.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

try:
    from .configuration_fetal_plane import FetalPlaneConfig
except ImportError:
    from configuration_fetal_plane import FetalPlaneConfig


class FetalPlaneForImageClassification(PreTrainedModel):
    config_class = FetalPlaneConfig

    def __init__(self, config: FetalPlaneConfig, class_weights: torch.Tensor = None):
        super().__init__(config)
        self.config = config

        self.features = nn.Sequential(
            # Input: (in_channels, resolution, resolution) -> default (1, 128, 128)
            nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # (64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(config.dropout_conv),
        )

        # Feature map size after 3 max-pools: 128 / (2^3) = 16
        spatial_dim = config.image_resolution // 8
        flattened_dim = 128 * spatial_dim * spatial_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_fc),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Initialize weights (standard HF method)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        features = self.features(pixel_values)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )