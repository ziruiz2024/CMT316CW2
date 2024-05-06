import tkinter as tk
import os
import os.path as op
import glob
import torch
import torch.nn as nn
import pandas as pd
import math
import numpy as np 
import xml.etree.ElementTree as ET
import pickle
from pathlib import Path
from tkinter import messagebox,ttk
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, 
)

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

        # Print the loss and accuracy
        print("Train Loss: ", np.mean(train_losses[-train_iters_per_epoch:]))
        print("Test Loss: ", np.mean(test_losses[-test_iters_per_epoch:]))
        print("Train Accuracy: ", train_accs[-1])
        print("Test Accuracy: ", test_accs[-1])
        
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
        test_accs=test_accs
    )

def plot_history(history, ax=None):
    if ax is None:
        ax = plt.gca()
    train_iters_per_epoch = history['train_iters_per_epoch']
    test_iters_per_epoch = history['test_iters_per_epoch']
    train_losses = history['train_losses']
    test_losses = history['test_losses']
    train_accs = history['train_accs']
    test_accs = history['test_accs']
    # Dual axis
    train_losses = np.array(train_losses)
    # Exponential moving average
    train_losses_smooth = np.zeros_like(train_losses)
    train_losses_smooth[0] = train_losses[0]
    for i in range(1, len(train_losses)):
        train_losses_smooth[i] = 0.9 * train_losses_smooth[i-1] + 0.1 * train_losses[i]
    test_losses = np.array(test_losses)
    test_losses_smooth = np.zeros_like(test_losses)
    test_losses_smooth[0] = test_losses[0]
    for i in range(1, len(test_losses)):
        test_losses_smooth[i] = 0.9 * test_losses_smooth[i-1] + 0.1 * test_losses[i]
    ax2 = ax.twinx()
    ax.plot(np.arange(len(train_losses)) / train_iters_per_epoch, train_losses, label="Train Loss", color='tab:blue', alpha=0.5)
    ax.plot(np.arange(len(test_losses)) / test_iters_per_epoch, test_losses, label="Test Loss", color='tab:orange', alpha=0.5)
    ax.plot(np.arange(len(train_losses)) / train_iters_per_epoch, train_losses_smooth, color='tab:blue')
    ax.plot(np.arange(len(test_losses)) / test_iters_per_epoch, test_losses_smooth, color='tab:orange')
    ax2.plot(np.arange(len(train_accs)), train_accs, label="Train Accuracy", color='tab:green')
    ax2.plot(np.arange(len(test_accs)), test_accs, label="Test Accuracy", color='tab:red')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid()
    return ax

def setup_gui():
    root = tk.Tk()
    root.title = "Model Selection"
    root.geometry("400x300")
    
    # Labels
    model_label = tk.Label(root, text="Select the model(s) to evaluate")
    batch_choices_label = tk.Label(root, text="Select the batch size(s) to include in model evaluation")

    # Dropdowns
    model_choices = ["resnet", "efficientnet", "vit", "twolayerscnn", "All"]
    batch_choices = ["32", "32,64", "32,64,128"]
    model_dropdown = ttk.Combobox(root, values=model_choices)
    batch_size_dropdown = ttk.Combobox(root, values=batch_choices)
    
    # Set default dropdown values
    model_dropdown.set(MODEL_CHOICE)
    batch_size_dropdown.set(BATCH_SIZE_CHOICE)

    # Packing
    model_label.pack(anchor='center', padx=10, pady=5)
    model_dropdown.pack(anchor='center', padx=10, pady=5)
    batch_choices_label.pack(anchor='center', padx=10, pady=5)
    batch_size_dropdown.pack(anchor='center', padx=10, pady=5)


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
    models_to_run = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "twolayerscnn"]
    batch_sizes = [32,64,128]
    
    if MODEL_CHOICE:
        if MODEL_CHOICE == "resnet":
            models_to_run = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        elif MODEL_CHOICE == "efficientnet":
            models_to_run = ["efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]
        elif MODEL_CHOICE == "vit":
            models_to_run = ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]
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
                if op.exists(path_history):
                    print("Model", name, "already trained, skipping")
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

            data.append(dict(
                model=model_name,
                batch_size=batch_size,
                train_acc_best="{:.4f}".format(max(history['train_accs'])),
                test_acc_best="{:.4f}".format(max(history['test_accs'])),
            ))

        result_df = pd.DataFrame(data)
        print(result_df.to_markdown())

        # Determine the grid size
        num_histories = len(histories)
        cols = 2  # You can adjust this number based on your display preferences
        rows = math.ceil(num_histories / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.5))  # Adjust height dynamically
        axes = axes.flatten()  # Flatten in case we have more than one row

        for i, hist in enumerate(histories):
            with open(hist, "rb") as f:
                history = pickle.load(f)
            name = op.basename(hist).replace(".pkl", "")
            ax = axes[i]
            plot_history(history, ax=ax)
            ax.set_title(name)
            

        plt.tight_layout()
        plt.show()
    else:
        print("No history files could be found for the selected model and batch size criteria")
    
if __name__ == '__main__':
    main()