import torch
import numpy as np
from transformers import SamModel, SamProcessor
from PIL import Image

class Segment_model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base", use_fast=True)

    def segment(self, image, points, labels=None):
        # Run segmentation with the given image, points, and labels
        if not points:
            return image

        if labels is None:
            labels = [1] * len(points)

        input_points = torch.tensor([points], dtype=torch.float).to(self.device)
        input_labels = torch.tensor([labels], dtype=torch.int).to(self.device)

        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )

        mask = masks[0][0,0].cpu().numpy()

        return mask