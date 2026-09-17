from transformers import AutoModelForImageClassification, AutoConfig
from PIL import Image
import torch
import torchvision.transforms as transforms

repo_id = "Marc-HealthAI/FP_Classification-V1"

# 1. Load model with remote custom code
model = AutoModelForImageClassification.from_pretrained(repo_id, trust_remote_code=True)
model.eval()

# 2. Preprocess an image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

img = Image.open("sample_fetal_plane.png")
pixel_values = transform(img).unsqueeze(0)  # Shape: (1, 1, 128, 128)

# 3. Predict
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    pred_idx = outputs.logits.argmax(-1).item()

print("Predicted Plane:", model.config.id2label[pred_idx])