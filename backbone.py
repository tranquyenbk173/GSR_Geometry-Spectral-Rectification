"""Frozen pre-trained backbones used as feature extractors."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

BACKBONES = ["dinov2", "mocov3"]


def get_processor(backbone_name: str):
    name = backbone_name.lower()
    if name == "dinov2":
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    if name == "mocov3":
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    raise ValueError(f"Unsupported backbone: {backbone_name} (choose from {BACKBONES})")


class Backbone:
    """ViT-B/16 (MoCo v3) or ViT-B/14 (DINOv2) returning the [CLS] feature."""

    num_features = 768

    def __init__(self, backbone_name: str, device, mocov3_ckpt: str = "mocov3-vit-base-300ep.pth"):
        self.name = backbone_name.lower()
        self.device = device

        if self.name == "dinov2":
            from transformers import AutoModel
            model = AutoModel.from_pretrained("facebook/dinov2-base")
        elif self.name == "mocov3":
            import timm
            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
            state = torch.load(mocov3_ckpt, map_location="cpu")["state_dict"]
            prefix = "module.base_encoder."
            state = {k[len(prefix):]: v for k, v in state.items()
                     if k.startswith(prefix) and not k[len(prefix):].startswith("head")}
            print("[MoCo v3]", model.load_state_dict(state, strict=False))
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name} (choose from {BACKBONES})")
        self.model = model.to(device).eval()

    @torch.no_grad()
    def forward(self, inputs) -> torch.Tensor:
        if self.name == "dinov2":
            pixel_values = inputs["pixel_values"][0].to(self.device)
            return self.model(pixel_values).last_hidden_state[:, 0]
        return self.model.forward_features(inputs.to(self.device))[:, 0]

    def get_features(self, dataloader):
        features, labels = [], []
        for images, targets in tqdm(dataloader):
            features.append(self.forward(images).cpu().numpy())
            labels.append(np.asarray(targets))
        return np.concatenate(features), np.concatenate(labels)
