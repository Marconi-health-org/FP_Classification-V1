import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoConfig,
    AutoModelForImageClassification,
)
from google.colab import userdata
from huggingface_hub import login
import wandb

# 1. Custom modules
from configuration_fetal_plane import FetalPlaneConfig
from modeling_fetal_plane import FetalPlaneForImageClassification

# Register custom architecture
AutoConfig.register("fetal_plane_cnn", FetalPlaneConfig)
AutoModelForImageClassification.register(FetalPlaneConfig, FetalPlaneForImageClassification)
FetalPlaneConfig.register_for_auto_class()
FetalPlaneForImageClassification.register_for_auto_class("AutoModelForImageClassification")

# 2. Authentication & Config
hf_token = userdata.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

repo_model_name = "Marc-HealthAI/FP_Classification-V1"

config_params = {
    "learning_rate": 1e-3,
    "epochs": 50,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "eval_batch_size": 16,
    "image_resolution": 128,
    "dataset": "Marc-HealthAI/fetal-planes-classification-dataset-main",
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,
}

wandb.init(project="Marc-HealthAI FP Classification", config=config_params, name=repo_model_name)

# 3. Load Dataset
dataset = load_dataset(config_params["dataset"])
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

class_names = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

# 4. Transforms and Datasets
res = config_params["image_resolution"]
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((res, res)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((res, res)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class FetalPlanesTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None):
        self.dataset = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("L")
        if self.transform:
            image = self.transform(image)
        return {
            "pixel_values": image,
            "labels": label2id[str(row["Plane"]).strip()],
        }

train_torch_ds = FetalPlanesTorchDataset(train_dataset, train_transform)
val_torch_ds = FetalPlanesTorchDataset(val_dataset, eval_transform)
test_torch_ds = FetalPlanesTorchDataset(test_dataset, eval_transform)

# 5. Class Weights
train_labels = np.asarray([label2id[str(lbl).strip()] for lbl in train_dataset["Plane"]])
weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=train_labels)
class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

# 6. Initialize Config & Model
model_config = FetalPlaneConfig(
    num_classes=len(class_names),
    in_channels=1,
    image_resolution=res,
    id2label=id2label,
    label2id=label2id,
)

model = FetalPlaneForImageClassification(
    config=model_config,
    class_weights=class_weights_tensor,
)

# 7. Compute Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

# 8. Training Arguments & Trainer
training_args = TrainingArguments(
    output_dir=repo_model_name,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=config_params["learning_rate"],
    lr_scheduler_type="cosine",
    per_device_train_batch_size=config_params["train_batch_size"],
    gradient_accumulation_steps=config_params["gradient_accumulation_steps"],
    per_device_eval_batch_size=config_params["eval_batch_size"],
    num_train_epochs=config_params["epochs"],
    logging_steps=10,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="recall_macro",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=True,
    hub_model_id=repo_model_name,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_torch_ds,
    eval_dataset=val_torch_ds,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=config_params["early_stopping_patience"],
            early_stopping_threshold=config_params["early_stopping_threshold"],
        )
    ],
)

# Train & Push
trainer.train()
trainer.push_to_hub()
wandb.finish()