import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import tkinter as tk
import plotly.graph_objects as go
import pickle
import pandas as pd
import os
import os.path as op
import numpy as np 
from tqdm import tqdm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, 
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tkinter import Checkbutton, ttk
from sklearn import metrics
from scipy.io import loadmat
from PIL import Image
from pathlib import Path

MODEL_CHOICE = "All"
BATCH_SIZE_CHOICE = "32,64,128"

def load_dataset():
    # Get train list
    f = loadmat("lists/train_list.mat")
    train_images = [x[0][0] for x in f['file_list']]
    train_labels = [x[0] for x in f['labels']]

    # Get file list
    f = loadmat("lists/test_list.mat")
    test_images = [x[0][0] for x in f['file_list']]
    test_labels = [x[0] for x in f['labels']]

    # Gather data
    train_data = []
    test_data = []

    # Record category ids
    categories = {}

    for i in range(len(train_images) + len(test_images)):

        # Determine if train or test
        image = train_images[i] if i < len(train_images) else test_images[i - len(train_images)]
        label = train_labels[i] if i < len(train_images) else test_labels[i - len(train_images)]
        label_name = os.path.split(image)[0]
        # Label -1 to make it 0-indexed
        categories[label_name] = label-1
        annotation_path = os.path.join("Annotation", image.replace(".jpg", ""))

        # Read XML
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        bndbox = root.find("object").find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Append to data
        if i < len(train_images):
            train_data.append(dict(
                image=os.path.join("Images", image),
                label=label-1,
                label_name=label_name,
                width=width,
                height=height,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            ))
        else:
            test_data.append(dict(
                image=os.path.join("Images", image),
                label=label-1,
                label_name=label_name,
                width=width,
                height=height,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            ))


    return train_data, test_data, categories

# Inherit from Dataset
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image'])
        image = image.convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        # Convert label to long data type
        label = torch.tensor(label, dtype=torch.long)
        return image, label


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


def train_model(model, device, train_dataset, test_dataset, batch_size=32):
    """
    Train the model, this function will return a dictionary containing the training and testing loss and accuracy
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3, verbose=True, factor=0.1)

    train_losses = []
    test_losses = []
    train_iters_per_epoch = len(train_dataloader)
    test_iters_per_epoch = len(test_dataloader)
    train_accs = []
    test_accs = []
    train_recall = []
    test_recall = []
    test_confusion_matrix = []
    train_confusion_matrix = []
    train_precision = []
    test_precision = []
    train_f1 = []
    test_f1 = []

    epoch = 1
    while True:

        model.train()
        for images, labels in tqdm(train_dataloader, desc="Train Epoch " + str(epoch)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc="Test Epoch " + str(epoch)):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        test_accs.append(metrics.accuracy_score(y_true, y_pred))
        test_recall.append(metrics.recall_score(y_true, y_pred, average='weighted', zero_division=1))
        test_precision.append(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=1))
        test_f1.append(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=1))
        test_confusion_matrix.append(metrics.confusion_matrix(y_true, y_pred))

        # Evaluate the model on training set
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in tqdm(train_dataloader, desc="Train Epoch " + str(epoch)):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        train_accs.append(metrics.accuracy_score(y_true, y_pred))
        train_recall.append(metrics.recall_score(y_true, y_pred, average='weighted', zero_division=1))
        train_precision.append(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=1))
        train_f1.append(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=1))
        train_confusion_matrix.append(metrics.confusion_matrix(y_true, y_pred))

        # Print the loss and accuracy
        print("Train Loss: ", np.mean(train_losses[-train_iters_per_epoch:]))
        print("Test Loss: ", np.mean(test_losses[-test_iters_per_epoch:]))
        print("Train Accuracy: ", train_accs[-1])
        print("Test Accuracy: ", test_accs[-1])
        print("Train Recall: ", train_recall[-1])
        print("Test Recall: ", test_recall[-1])
        print("Train Precision: ", train_precision[-1])
        print("Test Precision: ", test_precision[-1])
        print("Train F1-Score: ", train_f1[-1])
        print("Test F1-Score: ", test_f1[-1])
        print("Train Confusion Matrix: ", train_confusion_matrix[-1])
        print("Test Confusion Matrix: ", test_confusion_matrix[-1])
        
        scheduler.step(test_accs[-1])
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate is too low, stopping training")
            break
        epoch += 1

    return dict(
        train_iters_per_epoch=train_iters_per_epoch,
        test_iters_per_epoch=test_iters_per_epoch,
        train_losses=train_losses,
        test_losses=test_losses,
        train_accs=train_accs,
        test_accs=test_accs,
        train_recall=train_recall,
        test_recall=test_recall,
        train_precision=train_precision,
        test_precision=test_precision,
        train_f1=train_f1,
        test_f1=test_f1,
        train_confusion_matrix=train_confusion_matrix,
        test_confusion_matrix=test_confusion_matrix
    ),
    
def plot_metric_comparison(metrics_data, metric_name):
    if (metric_name == "accs"):
        display_name = "Accuracy"
    else:
        display_name = metric_name
    
    fig = go.Figure()
    for model_data in metrics_data["train_" + metric_name]:
        epochs = list(range(len(model_data['values'])))
        fig.add_trace(go.Scatter(x=epochs, y=model_data['values'], mode='lines+markers', name=model_data['model'].split("\\")[-1] + " (Train)"))
        
    for model_data in metrics_data["test_" + metric_name]:
        epochs = list(range(len(model_data['values'])))
        fig.add_trace(go.Scatter(x=epochs, y=model_data['values'], mode='lines+markers', name=model_data['model'].split("\\")[-1] + " (Test)"))
    
    fig.update_layout(
        title=f"{display_name.capitalize()} Comparison Across Models",
        xaxis_title="Epoch",
        yaxis_title= display_name.capitalize(),
        legend_title="Model Names",
        width=1280, 
        height=720 
    )
    fig.show()
    
def load_and_structure_histories(histories_directory, models_to_run, batch_sizes):
    all_histories = {}
    
    for model_name in models_to_run:
        for batch_size in batch_sizes:
            history_file = histories_directory + "\\" + f"{model_name}_{batch_size}.pkl"
            
            if (os.path.isfile(history_file)):
                model_name = history_file.replace(".pkl", "")
                with open(os.path.join(histories_directory, history_file), "rb") as f:
                    history = pickle.load(f)
                for key in history[0].keys():
                    if key not in all_histories:
                        all_histories[key] = []
                    all_histories[key].append({'model': model_name, 'values': history[0][key]})
                    
    return all_histories
    
def setup_gui():
    root = tk.Tk()
    root.title = "Model Selection"
    root.geometry("600x400")
    
    # Create checkbox variables
    global SHOW_LOSS_METRICS
    global SHOW_ACCURACY_METRICS
    global SHOW_PRECISION_METRICS
    global SHOW_RECALL_METRICS
    global SHOW_F1_METRICS
    
    SHOW_LOSS_METRICS = tk.BooleanVar(value=True)
    SHOW_ACCURACY_METRICS = tk.BooleanVar(value=True)
    SHOW_PRECISION_METRICS = tk.BooleanVar(value=True)
    SHOW_RECALL_METRICS = tk.BooleanVar(value=True)
    SHOW_F1_METRICS = tk.BooleanVar(value=True)
    
    # Labels
    model_label = tk.Label(root, text="Select the model(s) to evaluate")
    batch_choices_label = tk.Label(root, text="Select the batch size(s) to include in model evaluation")

    # Dropdowns
    model_choices = ["resnet", "efficientnet", "vit", "twolayerscnn", "All"]
    batch_choices = ["32", "32,64", "32,64,128"]
    model_dropdown = ttk.Combobox(root, values=model_choices)
    batch_size_dropdown = ttk.Combobox(root, values=batch_choices)
    
    # Checkboxes
    show_loss_metrics_checkbox = Checkbutton(root, text="Display Loss Metrics", variable=SHOW_LOSS_METRICS)
    show_accuracy_metrics_checkbox = Checkbutton(root, text="Display Accuracy Metrics", variable=SHOW_ACCURACY_METRICS)
    show_precision_metrics_checkbox = Checkbutton(root, text="Display Precision Metrics", variable=SHOW_PRECISION_METRICS)
    show_recall_metrics_checkbox = Checkbutton(root, text="Display Recall Metrics", variable=SHOW_RECALL_METRICS)
    show_f1_metrics_checkbox = Checkbutton(root, text="Display F1 Score Metrics", variable=SHOW_F1_METRICS)
    
    # Set default values
    model_dropdown.set(MODEL_CHOICE)
    batch_size_dropdown.set(BATCH_SIZE_CHOICE)
    show_loss_metrics_checkbox.select()
    show_accuracy_metrics_checkbox.select()
    show_precision_metrics_checkbox.select()
    show_recall_metrics_checkbox.select()
    show_f1_metrics_checkbox.select()

    # Packing
    model_label.pack(anchor='center', padx=10, pady=5)
    model_dropdown.pack(anchor='center', padx=10, pady=5)
    batch_choices_label.pack(anchor='center', padx=10, pady=5)
    batch_size_dropdown.pack(anchor='center', padx=10, pady=5)
    show_loss_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_accuracy_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_precision_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_recall_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_f1_metrics_checkbox.pack(anchor='center', padx=10, pady=5)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit_click(model_dropdown, batch_size_dropdown, root))
    submit_button.pack(anchor='center', padx=10, pady=20)

    # Keep the main loop running until explicitly destroyed
    root.mainloop()

def on_submit_click(model_dropdown, batch_size_dropdown, root):
    global MODEL_CHOICE, BATCH_SIZE_CHOICE
    
    MODEL_CHOICE = model_dropdown.get()
    BATCH_SIZE_CHOICE = batch_size_dropdown.get()
    
    # Postpone root destruction to after the main loop
    root.after(100, root.destroy)
    

def main():
    setup_gui()
    models_to_run = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3" 
                     "efficientnet_b4", "vit_b_16", "vit_b_32", "twolayerscnn"]
    batch_sizes = [32,64,128]
    
    if MODEL_CHOICE:
        if MODEL_CHOICE == "resnet":
            models_to_run = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        elif MODEL_CHOICE == "efficientnet":
            models_to_run = [ "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"]
        elif MODEL_CHOICE == "vit":
            models_to_run = ["vit_b_16", "vit_b_32"]
        elif MODEL_CHOICE == "twolayerscnn":
            models_to_run = ["twolayerscnn"]
            
    if BATCH_SIZE_CHOICE:
        if BATCH_SIZE_CHOICE == "32":
            batch_sizes = [32]
        elif BATCH_SIZE_CHOICE == "32,64":
            batch_sizes = [32,64]
        elif BATCH_SIZE_CHOICE == "32,64,128":
            batch_sizes = [32,64,128]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read dataset and gather into dataframe
    train_data, test_data, categories = load_dataset()
    dftrain = pd.DataFrame(train_data)
    dftest = pd.DataFrame(test_data)

    # Get the classes summary
    print("Number of classes: ", len(categories))
    print("Number of training samples: ", len(dftrain))
    print("Number of testing samples: ", len(dftest))

    train_transforms = transforms.Compose([
        # Randomly resize and crop the image to 224
        transforms.RandomResizedCrop(224),
        # Randomly flip the image horizontally
        transforms.RandomHorizontalFlip(),
        # Convert the image to a PyTorch Tensor
        transforms.ToTensor(),
        # Normalize the image
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        # Resize the image to 256
        transforms.Resize(256),
        # Crop the center of the image
        transforms.CenterCrop(224),
        # Convert the image to a PyTorch Tensor
        transforms.ToTensor(),
        # Normalize the image
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(dftrain, transform=train_transforms)
    test_dataset = CustomDataset(dftest, transform=test_transforms)

    for model_name in models_to_run:
        for batch_size in batch_sizes:
            print(model_name + "_" + str(batch_size))
            try:
                name = model_name + "_" + str(batch_size)
                path_history = op.join("Histories", name + ".pkl")
                path_model = op.join("Histories", name + ".pth")
                
                # Skip models that have already been built
                if op.exists(path_history):
                    print("Model", name, "already trained, skipping")
                    continue
                
                # Skip combinations of model_name and batch_size that cause an out of memory exception to throw
                if ((model_name in ["resnet50", "efficientnet_b0", "efficientnet_b1", "twolayerscnn"] and batch_size == 128) or 
                    (model_name in ["resnet101", "resnet152", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "vit_b_16"] and batch_size in [64, 128])):
                    print(f"Skipping training for {model_name} with batch size {batch_size} to avoid OOM error")
                    continue
                model = build_model(model_name, categories).to(device)
                history = train_model(model, device, train_dataset, test_dataset, batch_size=batch_size)
                with open(path_history, "wb") as f:
                    pickle.dump(history, f)
                torch.save(model.state_dict(), path_model)
                print("Model", name, "trained and saved")
            # OOM
            except RuntimeError as e:
                print(str(e))
                print("Model", name, "OOM, skipping")
            except Exception as e:
                print(str(e))
                print("Model", name, "Error", e)
                continue

    histories = []
    base_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    for model_name in models_to_run:
        for batch_size in batch_sizes:
            history_file = base_directory / "Histories" / f"{model_name}_{batch_size}.pkl"
            
            if (history_file.is_file()):
                histories.append(history_file)
    
    if (len(histories) > 0):
        data = []
        for hist in histories:
            with open(hist, "rb") as f:
                history = pickle.load(f)
            model_name = op.basename(hist).replace(".pkl", "")
            model_name_list = model_name.split("_")
            model_name = "_".join(model_name_list[:-1])
            batch_size = int(model_name_list[-1])
            print(type(history))
            data.append(dict(
                model=model_name,
                batch_size=batch_size,
                train_acc_best="{:.4f}".format(max(history[0]['train_accs'])),
                test_acc_best="{:.4f}".format(max(history[0]['test_accs'])),
            ))

        all_histories = load_and_structure_histories(str(base_directory) + "\\Histories", models_to_run, batch_sizes)
        metrics = []
        
        if (SHOW_LOSS_METRICS.get()):
            metrics.append("loss")
        if (SHOW_ACCURACY_METRICS.get()):
            metrics.append("accs")
        if (SHOW_PRECISION_METRICS.get()):
            metrics.append("precision")
        if (SHOW_RECALL_METRICS.get()):
            metrics.append("recall")
        if (SHOW_F1_METRICS.get()):
            metrics.append("f1")
        
        for metric in metrics:
            if "test_" + metric in all_histories:
                # Plot the training metric
                plot_metric_comparison(all_histories, metric)

    else:
        print("No history files could be found for the selected model and batch size criteria")
    
if __name__ == '__main__':
    main()