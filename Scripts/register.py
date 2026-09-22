import os
import sys
import shutil
import subprocess
import torch
from google.colab import userdata

# 1. Resolve imports from scripts/ or root
if os.path.exists("scripts"):
    sys.path.append("scripts")
elif os.path.exists("Scripts"):
    sys.path.append("Scripts")

from configuration_fetal_plane import FetalPlaneConfig
from modeling_fetal_plane import FetalPlaneForImageClassification
from transformers import AutoConfig, AutoModelForImageClassification

# 2. Register Custom AutoClasses
AutoConfig.register("fetal_plane_cnn", FetalPlaneConfig)
AutoModelForImageClassification.register(FetalPlaneConfig, FetalPlaneForImageClassification)

FetalPlaneConfig.register_for_auto_class()
FetalPlaneForImageClassification.register_for_auto_class("AutoModelForImageClassification")

# 3. Define Labels & Config
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

config = FetalPlaneConfig(
    num_classes=len(class_names),
    in_channels=1,
    image_resolution=128,
    id2label=id2label,
    label2id=label2id,
)

# 4. Initialize model architecture
model = FetalPlaneForImageClassification(config=config)

# 5. Load your existing trained weights
model_folder = "FP_Classification-V1"

# Look for safetensors or pytorch_model.bin in the existing folder or checkpoints
safetensors_path = os.path.join(model_folder, "model.safetensors")
bin_path = os.path.join(model_folder, "pytorch_model.bin")

if os.path.exists(safetensors_path):
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)
    print("✓ Loaded weights from model.safetensors")
elif os.path.exists(bin_path):
    state_dict = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print("✓ Loaded weights from pytorch_model.bin")
else:
    print("⚠ No local weights file found in", model_folder)
    print("If you already have weights on Hugging Face, we will save config & code structure.")

# 6. Save model (this injects 'auto_map' into config.json automatically)
model.save_pretrained(model_folder)
print(f"✓ Saved updated model and auto_map config to '{model_folder}'")

# 7. Copy architecture scripts into the model folder
for script_name in ["configuration_fetal_plane.py", "modeling_fetal_plane.py"]:
    for search_dir in ["scripts", "Scripts", "."]:
        src = os.path.join(search_dir, script_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(model_folder, script_name))
            print(f"✓ Copied {script_name} into {model_folder}/")
            break

# 8. Push updated files to GitHub (which triggers the GitHub Action to sync to HF)
branch = "colab"
github_token = userdata.get("GH_TOKEN")
github_repo_url = userdata.get("REPO_URL")
git_email = userdata.get("Email")
git_name = userdata.get("NAME")

if github_token and github_repo_url:
    auth_repo_url = github_repo_url.replace("https://", f"https://{github_token}@")

    push_script = f"""
    cd {model_folder}
    if [ ! -d ".git" ]; then
        git init
        git remote add origin {auth_repo_url}
    fi
    git remote set-url origin {auth_repo_url}
    git config user.email "{git_email}"
    git config user.name "{git_name}"
    git checkout -B {branch}
    git add .
    git commit -m "Register custom architecture and auto_map without retraining" || echo "No changes to commit"
    git push -u origin {branch} --force
    """

    res = subprocess.run(push_script, shell=True, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.returncode == 0:
        print(f"✓ Successfully pushed registered model files to GitHub branch '{branch}'!")
    else:
        print("STDERR:\n", res.stderr)