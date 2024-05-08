import torch.nn as nn
import torch
import tkinter as tk
from scipy.io import loadmat
import requests
import os
import json
import glob
import gdown
from tqdm import tqdm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, 
)
from torchvision import transforms
from tkinter import ttk
from PIL import Image
from flask import Flask, request, jsonify
from PIL import Image
import io

BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Use efficientnet_b4_32 by default as this is the best performing model
MODEL_CHOICE =   "efficientnet_b4_32"

def build_model(model_name, categories):
    if model_name == 'twolayerscnn':
        model = TwoLayersCNN(len(categories))
    elif model_name == 'resnet18':
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
    elif model_name == 'resnet34':
        model = resnet34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
    elif model_name == 'resnet50':
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
    elif model_name == 'resnet101':
        model = resnet101(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
    elif model_name == 'resnet152':
        model = resnet152(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=True)
        classifier = model.classifier
        dropoutrate = classifier[0].p
        infeatures = classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropoutrate),
            nn.Linear(infeatures, len(categories))
        )
    elif model_name == 'efficientnet_b1':
        model = efficientnet_b1(pretrained=True)
        classifier = model.classifier
        dropoutrate = classifier[0].p
        infeatures = classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropoutrate),
            nn.Linear(infeatures, len(categories))
        )
    elif model_name == 'efficientnet_b2':
        model = efficientnet_b2(pretrained=True)
        classifier = model.classifier
        dropoutrate = classifier[0].p
        infeatures = classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropoutrate),
            nn.Linear(infeatures, len(categories))
        )
    elif model_name == 'efficientnet_b3':
        model = efficientnet_b3(pretrained=True)
        classifier = model.classifier
        dropoutrate = classifier[0].p
        infeatures = classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropoutrate),
            nn.Linear(infeatures, len(categories))
        )
    elif model_name == 'efficientnet_b4':
        model = efficientnet_b4(pretrained=True)
        classifier = model.classifier
        dropoutrate = classifier[0].p
        infeatures = classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropoutrate),
            nn.Linear(infeatures, len(categories))
        )
    elif model_name == 'vit_b_16':
        model = vit_b_16(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(categories))
    elif model_name == 'vit_b_32':
        model = vit_b_32(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(categories))
    elif model_name == 'vit_l_16':
        model = vit_l_16(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(categories))
    elif model_name == 'vit_l_32':
        model = vit_l_32(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(categories))
    else:
        raise ValueError("Model not supported")
    
    return model

class TwoLayersCNN(nn.Module):
    def __init__(self, num_classes):
        super(TwoLayersCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_categories():
    # Get train list
    f = loadmat(os.path.join(BASE_DIRECTORY, "lists", "train_list.mat"))
    train_images = [x[0][0] for x in f['file_list']]
    train_labels = [x[0] for x in f['labels']]

    # Get file list
    f = loadmat(os.path.join(BASE_DIRECTORY, "lists", "test_list.mat"))
    test_images = [x[0][0] for x in f['file_list']]
    test_labels = [x[0] for x in f['labels']]

    # Record category ids
    categories = {}

    for i in range(len(train_images) + len(test_images)):
        # Determine if train or test
        image = train_images[i] if i < len(train_images) else test_images[i - len(train_images)]
        label = train_labels[i] if i < len(train_images) else test_labels[i - len(train_images)]
        label_name = os.path.split(image)[0]
        # Label -1 to make it 0-indexed
        categories[label_name] = label-1

    return categories

# Load the model
def load_model():
    if 'model_choice' in os.environ:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.environ['model_path']
        model_to_use = os.environ['model_choice']
        print(f"Using model  {model_to_use} for classifier")
        
        # Some of the .pth files are the full PyTorch snapshot and some are simply dictionary state files. Need to dynamically handle this.
        # First, attempt to load the full model
        model = torch.load(model_path, map_location=DEVICE)
        if isinstance(model, torch.nn.Module):
            print("Loaded full model")
        else:
            # If not a full model, assume it's a state dict and load accordingly
            print("Loaded model is a state dict, loading into appropriate model architecture.")
            # Assume it's a state dict if direct loading failed
            model = build_model(model_to_use, load_categories())
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()
    return model, DEVICE


# Image processing
def transform_image(image_bytes):
     # Convert the binary data to a bytes or BytesIO object
    image = Image.open(io.BytesIO(image_bytes))
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = my_transforms(image).unsqueeze(0)
    
    return tensor.to(DEVICE)

app = Flask(__name__)
app.config['DEBUG'] = False  # Explicitly set debug to False

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        with torch.no_grad():
            outputs = MODEL(tensor)
            _, predicted = outputs.max(1)
            # Map the prediction index to the corresponding class name
            prediction_class = INDEX_TO_CLASS[predicted.item()].split('-')[-1]
            response = {'prediction': prediction_class}
            return jsonify(response)
        
def download_pre_built_models(histories_directory, model_file_path):
    is_download_successful = False
    
    pre_built_model_links = json.loads(json.dumps({"efficientnet_b0_32.pth":"https://drive.google.com/uc?export=download&id=1F4tOSJfweuBEGtSzMMpBPbJmtHXY2sp9",
                                                    "efficientnet_b0_64.pth":"https://drive.google.com/uc?export=download&id=1xn7GGCisTdU6laGMqxtbLWNZuw4bQQyy",
                                                    "efficientnet_b1_32.pth":"https://drive.google.com/uc?export=download&id=1E7h3b2ciShT4t47y6Mp12S4W4z55sTnK",
                                                    "efficientnet_b4_32.pth":"https://drive.google.com/uc?export=download&id=1Q0J29a7tkT2rcDDrEbKguNJDg71dnUYW",
                                                    "resnet18_32.pth":"https://drive.google.com/uc?export=download&id=1ors9Gj3ZRs9JJGcPfMyjU5qCHBcxVgRC",
                                                    "resnet18_64.pth":"https://drive.google.com/uc?export=download&id=1Pw-DoXr_GsxNuguZC5UcQbH9q3my1q2w",
                                                    "resnet18_128.pth":"https://drive.google.com/uc?export=download&id=1QK9ooDuWytklJNyeRi4-uTaO-9m8gvlX",
                                                    "resnet34_32.pth":"https://drive.google.com/uc?export=download&id=1BkxV3Gw0bUFLCwAG30Mb_fAi5b6iMa4p",
                                                    "resnet34_64.pth":"https://drive.google.com/uc?export=download&id=1kdfLlvO_lVnnydRnCwqG09-LJxUr22UR",
                                                    "resnet34_128.pth":"https://drive.google.com/uc?export=download&id=1kSGFiSiFJ1Z6T1IxaeVc2jUcpAsLpPmb",
                                                    "resnet50_32.pth":"https://drive.google.com/uc?export=download&id=1NEfvLrhHnd_JqJXmJyUtUYnEhjI4WzG5",
                                                    "resnet50_64.pth":"https://drive.google.com/uc?export=download&id=1Pw-DoXr_GsxNuguZC5UcQbH9q3my1q2w",
                                                    "resnet101_32.pth":"https://drive.google.com/uc?export=download&id=1BhFIBX4d5sgnQewkPYeRN3SSod_QXoFb",
                                                    "resnet152_32.pth":"https://drive.google.com/uc?export=download&id=1etFBqHK1oiFpQdkKTL29vBR_nsiCaTbX"}))
  
    model_file_name = os.path.basename(model_file_path)
    
    if (os.path.isfile(model_file_path)):
        print(f"{model_file_name} already exists, no need to download")
        is_download_successful = True
    else:
        if (model_file_name in pre_built_model_links):
            pre_built_modeld_link = pre_built_model_links[model_file_name]
            print(f"{model_file_name} was not found , it will be downloaded from: {pre_built_modeld_link}.")
            
            gdown.download(pre_built_modeld_link, model_file_path)
                        
            is_download_successful = True
        else:
                print(f"{model_file_name} was not found , and a pre built downloaded could not be found for it.")
        
    return is_download_successful
        
def setup_gui():
    root = tk.Tk()
    root.title = "Model Selection"
    root.geometry("300x200")
    
    model_choices = ["resnet18_32", "resnet18_64", "resnet18_128", "resnet34_32", "resnet34_64", "resnet34_128", "resnet50_32", "resnet50_64", "resnet101_32", 
                     "resnet152_32", "efficientnet_b0_32", "efficientnet_b0_64", "efficientnet_b1_32", "efficientnet_b1_64", "efficientnet_b2_32", "efficientnet_b3_32", 
                     "efficientnet_b4_32", "vit_b_16_32", "vit_b_32_32", "vit_b_32_64", "vit_b_32_128", "twolayerscnn_32", "twolayerscnn_64"]
    
    # Labels
    model_label = tk.Label(root, text="Select the model to use for classification")
    
    model_dropdown = ttk.Combobox(root, values=model_choices)
    
    # Set default values
    model_dropdown.set(MODEL_CHOICE)

    # Packing
    model_label.pack(anchor='center', padx=10, pady=5)
    model_dropdown.pack(anchor='center', padx=10, pady=5)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit_click(root, model_dropdown))
    submit_button.pack(anchor='center', padx=10, pady=20)

    # Keep the main loop running until explicitly destroyed
    root.mainloop()

def on_submit_click(root, model_dropdown):
    # Strip off the batch suffix from the model name
    model_name = model_dropdown.get()
    model_name = model_name[:(model_name.rfind('_'))]
    os.environ['model_choice'] = model_name
    os.environ['model_path'] = os.path.join(BASE_DIRECTORY, "Histories", f"{model_dropdown.get()}.pth")
    
    # Ensure model is available
    if (download_pre_built_models(os.path.join(BASE_DIRECTORY, "Histories"), f"{os.environ['model_path']}")):
        root.withdraw()
        MODEL, DEVICE = load_model()
        app.run(use_reloader=True)
    else:
        print("Model not found and program was unable to download pre-built model")

def main():
    global MODEL, DEVICE, INDEX_TO_CLASS
    
    if 'WERKZEUG_LOADED' in os.environ:
        print("Reloading...")
        categories = load_categories()
        INDEX_TO_CLASS = {v: k for k, v in categories.items()}
        MODEL, DEVICE = load_model()        
        app.run(use_reloader=True)
    else:
        print("Starting...")
        os.environ['WERKZEUG_LOADED']='TRUE'
        setup_gui()
        
        categories = load_categories()
        INDEX_TO_CLASS = {v: k for k, v in categories.items()}
        MODEL, DEVICE = load_model()        
        app.run(use_reloader=True)

if __name__ == '__main__':
    main()