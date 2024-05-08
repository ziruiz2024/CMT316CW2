from scipy.io import loadmat
import torch
import tkinter as tk
import os
import glob
from torchvision.models import resnet152
from torchvision import transforms
from tkinter import ttk
from PIL import Image
from flask import Flask, request, jsonify
from PIL import Image
import io

BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Use efficientnet_b4_32 by default as this is the best performing model
MODEL_CHOICE =   "efficientnet_b4_32.pth"

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
        print(f"Loading model with {os.environ['model_choice']}")
        model_to_use = os.environ['model_choice']
        print(f"Using model  {model_to_use} for classifier")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL = torch.load(os.path.join(BASE_DIRECTORY, "Histories", model_to_use), map_location=DEVICE)
    return MODEL, DEVICE

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
        
def setup_gui():
    root = tk.Tk()
    root.title = "Model Selection"
    root.geometry("300x200")
    
    models = glob.glob(os.path.join(BASE_DIRECTORY, "Histories/*.pth"))
    model_choices = []
    
    # Labels
    model_label = tk.Label(root, text="Select the model to use for classification")
    
    # Dropdowns
    for model in models:
        model_choices.append(os.path.basename(model))
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
    os.environ['model_choice'] = os.path.join(BASE_DIRECTORY, "Histories", model_dropdown.get())
    root.withdraw()
    
    MODEL, DEVICE = load_model()
    app.run(use_reloader=True)

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