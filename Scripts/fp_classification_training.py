
"""FP CLASSIFICATION TRAINING"""

import os
import sys
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from huggingface_hub import login
from google.colab import userdata

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoConfig,
    AutoModelForImageClassification,
)


# Custom Modules & HF Auto-Class Registration

if os.path.exists("scripts"):
    sys.path.append("scripts")
elif os.path.exists("Scripts"):
    sys.path.append("Scripts")

from configuration_fetal_plane import FetalPlaneConfig
from modeling_fetal_plane import FetalPlaneForImageClassification

# Register architecture with AutoClasses
AutoConfig.register("fetal_plane_cnn", FetalPlaneConfig)
AutoModelForImageClassification.register(FetalPlaneConfig, FetalPlaneForImageClassification)

# Register for dynamic code packaging
FetalPlaneConfig.register_for_auto_class()
FetalPlaneForImageClassification.register_for_auto_class("AutoModelForImageClassification")


# Authentication & Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

wandb_key = userdata.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)

hf_token = userdata.get("HF_TOKEN") or userdata.get("HF_MARC")
if hf_token:
    login(token=hf_token)

repo_model_name = "FP_Classification-V1"

config_params = {
    "learning_rate": 1e-3,
    "epochs": 50,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "eval_batch_size": 16,
    "image_resolution": 128,
    "architecture": "CNN-Pytorch",
    "dataset": "Marc-HealthAI/fetal-planes-classification-dataset-main",
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,
}

wandb.init(
    project="Marc-HealthAI FP Classification",
    config=config_params,
    name=repo_model_name,
)


#Dataset & Label Mapping

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


#Transforms and Datasets

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


#Class Weights & Model Initialization

train_labels = np.asarray([label2id[str(lbl).strip()] for lbl in train_dataset["Plane"]])
weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=train_labels)
class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

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


# 6. Evaluation Metrics

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


# Training Setup

training_args = TrainingArguments(
    output_dir=repo_model_name,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=config_params["learning_rate"],
    lr_scheduler_type="cosine",
    dataloader_num_workers=2,
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
    push_to_hub=False,              # Disabled direct push to HF Hub
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


# Train

train_result = trainer.train()

print("Training complete.")
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best metric:", trainer.state.best_metric)


# Final Test Evaluation

test_output = trainer.predict(test_torch_ds)
test_logits = test_output.predictions
test_labels = test_output.label_ids
test_predictions = np.argmax(test_logits, axis=-1)

test_accuracy = accuracy_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)
precision_macro, recall_macro, _, _ = precision_recall_fscore_support(test_labels, test_predictions, average="macro", zero_division=0)

print(f"Test Accuracy        : {test_accuracy:.4f}")
print(f"Test F1 Macro        : {test_f1:.4f}")
print(f"Test Precision Macro : {precision_macro:.4f}")
print(f"Test Recall Macro    : {recall_macro:.4f}")

report_str = classification_report(test_labels, test_predictions, target_names=class_names, digits=4, zero_division=0)
print("\nClassification Report:\n", report_str)

# Save artifacts
os.makedirs(repo_model_name, exist_ok=True)
with open(os.path.join(repo_model_name, "classification_report.txt"), "w") as f:
    f.write(report_str)

cm = confusion_matrix(test_labels, test_predictions)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(repo_model_name, "confusion_matrix.csv"), index=True)

# Plot & save Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix — Fetal Plane Classification")
plt.tight_layout()
plt.savefig(os.path.join(repo_model_name, "confusion_matrix.png"))
plt.close()

# Log to W&B
wandb.log({
    "test_accuracy": test_accuracy,
    "test_f1_macro": test_f1,
    "test_precision_macro": precision_macro,
    "test_recall_macro": recall_macro,
    "confusion_matrix": wandb.plot.confusion_matrix(preds=test_predictions, y_true=test_labels, class_names=class_names),
})
wandb.finish()


#  Save Model & Prepare for GitHub Sync

branch = "colab"

# Save the model and tokenizer/config
trainer.save_model(repo_model_name)

# Ensure architecture files are copied into the model output folder
for script_name in ["configuration_fetal_plane.py", "modeling_fetal_plane.py"]:
    for search_dir in ["scripts", "Scripts", "."]:
        src = os.path.join(search_dir, script_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(repo_model_name, script_name))
            break

# Update README if dynamic sections exist
generated_readme_path = os.path.join(repo_model_name, "README.md")
eval_summary_content = ""
training_results_content = ""

if os.path.exists(generated_readme_path):
    with open(generated_readme_path, "r", encoding="utf-8") as f:
        gen_text = f.read()

    eval_match = re.search(r"(It achieves the following results on the evaluation set:\s*\n)([\s\S]*?)(?=\n## Model description)", gen_text)
    if eval_match:
        eval_summary_content = eval_match.group(2).strip()

    train_match = re.search(r"(### Training hyperparameters[\s\S]*?)(?=\n### Framework versions)", gen_text)
    if train_match:
        training_results_content = train_match.group(1).strip()

# Cleanup checkpoints
for item in os.listdir(repo_model_name):
    if item.startswith("checkpoint-"):
        shutil.rmtree(os.path.join(repo_model_name, item))

with open(os.path.join(repo_model_name, ".gitignore"), "w") as f:
    f.write("checkpoint-*/\nruns/\n")


# Git Push to GitHub

github_token = userdata.get("GH_TOKEN")
github_repo_url = userdata.get("REPO_URL")
git_email = userdata.get("Email")
git_name = userdata.get("NAME")

auth_repo_url = github_repo_url.replace("https://", f"https://{github_token}@")

push_script = f"""
cd {repo_model_name}
if [ ! -d ".git" ]; then
    git init
    git remote add origin {auth_repo_url}
fi
git remote set-url origin {auth_repo_url}
git config user.email "{git_email}"
git config user.name "{git_name}"
git checkout -B {branch}

git fetch origin {branch} 2>/dev/null || true
git checkout origin/{branch} -- README.md 2>/dev/null || true
rm -rf checkpoint-*
"""
subprocess.run(push_script, shell=True, capture_output=True)

# Inject results into custom README sections if present
target_readme_path = os.path.join(repo_model_name, "README.md")
if os.path.exists(target_readme_path):
    with open(target_readme_path, "r", encoding="utf-8") as f:
        readme_data = f.read()

    if eval_summary_content and "<!-- START_EVAL_SUMMARY -->" in readme_data:
        readme_data = re.sub(
            r"<!-- START_EVAL_SUMMARY -->[\s\S]*?<!-- END_EVAL_SUMMARY -->",
            f"<!-- START_EVAL_SUMMARY -->\n{eval_summary_content}\n<!-- END_EVAL_SUMMARY -->",
            readme_data
        )

    if training_results_content and "<!-- START_TRAINING_RESULTS -->" in readme_data:
        readme_data = re.sub(
            r"<!-- START_TRAINING_RESULTS -->[\s\S]*?<!-- END_TRAINING_RESULTS -->",
            f"<!-- START_TRAINING_RESULTS -->\n{training_results_content}\n<!-- END_TRAINING_RESULTS -->",
            readme_data
        )

    with open(target_readme_path, "w", encoding="utf-8") as f:
        f.write(readme_data)

# Push all files to GitHub
final_push_script = f"""
cd {repo_model_name}
git rm -rf checkpoint-* 2>/dev/null || true
git add .
git commit -m "Auto-update training artifacts on {branch}" || echo "No changes to commit"
git push -u origin {branch} --force
"""

result = subprocess.run(final_push_script, shell=True, capture_output=True, text=True)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

if result.returncode == 0:
    print(f"Successfully updated repository and pushed to branch '{branch}'!")
else:
    print("Check error logs above for details.")