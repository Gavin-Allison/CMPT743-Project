import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()
)

class AdaIN:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # full vgg loaded for weights; we'll use only part as encoder
        full_vgg = vgg.to(self.device)
        self.decoder = decoder.to(self.device)
        # encoder is first several layers (up to relu4_1)
        # original AdaIN uses VGG-19 layers 0:31
        encoder_layers = list(full_vgg.children())[:31]
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        # keep full_vgg around for weight-loading convenience
        self.vgg = full_vgg

        # load weights relative to this file
        base = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.normpath(os.path.join(base, "..", "models"))
        vgg_path = os.path.join(model_dir, "vgg_normalised.pth")
        dec_path = os.path.join(model_dir, "decoder.pth")

        state = torch.load(vgg_path, map_location=self.device)
        self.vgg.load_state_dict(state)
                
        state = torch.load(dec_path, map_location=self.device)
        self.decoder.load_state_dict(state)

        self.vgg.eval()
        self.decoder.eval()

    def adain(self, content_feat, style_feat, eps=1e-5):
        c_mean = content_feat.mean(dim=[2,3], keepdim=True)
        c_std = content_feat.std(dim=[2,3], keepdim=True) + eps
        s_mean = style_feat.mean(dim=[2,3], keepdim=True)
        s_std = style_feat.std(dim=[2,3], keepdim=True) + eps
        
        normalized = (content_feat - c_mean) / c_std
        return normalized * s_std + s_mean

    # Image preprocessing
    def transform(self, img, size=512):
        # Resize and convert to tensor normalized for VGG
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        return transform(img).unsqueeze(0)

    def stylize(self, content, style, alpha=1.0):     
        content = self.transform(content).to(self.device)
        style = self.transform(style).to(self.device)

        with torch.no_grad():
            # encode only first layers for AdaIN
            content_feat = self.encoder(content)
            style_feat = self.encoder(style)
            t = alpha * self.adain(content_feat, style_feat) + (1 - alpha) * content_feat
            out = self.decoder(t)
        out = out.squeeze(0)
       
        # decoder output uses normalized space; undo ImageNet normalization
        mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(3,1,1)
        std = torch.tensor([0.229,0.224,0.225], device=self.device).view(3,1,1)
        out = out * std + mean
        
        return out.clamp(0, 1)
    
