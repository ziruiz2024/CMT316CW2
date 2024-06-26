"""
Image Classification and Model Training System

This module provides functionalities for training and evaluating various machine learning models on dog breed image classification.
It supports a range of architectures from simpler convolutional networks to more complex models like ResNet and EfficientNet.

Features include:
- Loading and preprocessing datasets from .mat files and XML annotations.
- Defining custom datasets and dataloaders for image handling.
- Building and training predefined model architectures with custom output layers to fit the number of target categories.
- GUI for model and batch size selection to streamline the evaluation process.
- Tracking and saving performance metrics such as loss, accuracy, precision, recall, and F1 score.

The script requires PyTorch, torchvision, pandas, numpy, and tkinter among other libraries. It is structured to run from
the command line, initializing with a GUI for setup before commencing the training cycle. The models and their training
histories are saved in specified directories for later analysis.

Usage:
    python model_builder.py

Dependencies:
    PyTorch, torchvision, tkinter, pandas, numpy, tqdm, sklearn, scipy, PIL
"""
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import tkinter as tk
import pickle
import pandas as pd
import os
import os.path as op
import numpy as np 
import math
import glob
from tqdm import tqdm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, 
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tkinter import ttk, Checkbutton
from sklearn import metrics
from scipy.io import loadmat
from PIL import Image
from matplotlib import pyplot as plt

# Declare global variables
MODEL_CHOICE = "All"
BATCH_SIZE_CHOICE = "32,64,128"
BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def load_dataset():
    """
    Loads image data and annotations from a dataset, separating the data into training and testing sets.

    This function performs the following operations:
    - Loads a list of image filenames and labels from 'train_list.mat' and 'test_list.mat' files located within the specified base directory.
    - Constructs the training and testing datasets by reading the associated XML files for each image to obtain bounding box annotations and other metadata.
    - Records the categorical IDs for each label based on the directory names in the dataset, adjusting labels to be zero-indexed.

    Each entry in the training and testing datasets is a dictionary containing:
    - The path to the image
    - The zero-indexed label ID
    - The label name (directory name)
    - Image dimensions (width and height)
    - Bounding box coordinates (xmin, ymin, xmax, ymax)

    Returns:
        tuple: A tuple containing:
            - List of dictionaries for training data
            - List of dictionaries for testing data
            - Dictionary mapping label names to zero-indexed label IDs
    """
    
    # Get train list
    f = loadmat(os.path.join(BASE_DIRECTORY, "lists", "train_list.mat"))
    train_images = [x[0][0] for x in f['file_list']]
    train_labels = [x[0] for x in f['labels']]

    # Get file list
    f = loadmat(os.path.join(BASE_DIRECTORY, "lists", "test_list.mat"))
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
        annotation_path = os.path.join(BASE_DIRECTORY, "Annotation", image.replace(".jpg", ""))

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
    """
    A custom dataset class that extends PyTorch's Dataset class. This class is tailored for handling image datasets.
    
    Attributes:
        df (DataFrame): A pandas DataFrame containing the paths to the images and their corresponding labels.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    """
    
    def __init__(self, df, transform=None):
        """
        Initializes the CustomDataset instance with a DataFrame and optional transform.

        Parameters:
            df (DataFrame): The DataFrame containing the image paths and labels.
            transform (callable, optional): The transform to be applied to each image.
        """
        
        self.df = df
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: The total number of images.
        """
        
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Retrieves an image and its label from the dataset at the specified index. Applies a transform to the image if one is provided.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its label as a tensor.
        """
        
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
    """
    A simple two-layer convolutional neural network that extends PyTorch's nn.Module.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer with a kernel size of 3 and padding of 1, outputting 64 feature maps.
        conv2 (nn.Conv2d): The second convolutional layer with a kernel size of 3 and padding of 1, outputting 128 feature maps.
        pool (nn.MaxPool2d): Max pooling layer that reduces spatial dimensions by half.
        fc1 (nn.Linear): A fully connected layer that flattens the output of the second convolutional layer and feeds it into 512 neurons.
        fc2 (nn.Linear): The final fully connected layer that outputs the logits for each class.
    """
    
    def __init__(self, num_classes):
        """
        Initializes the TwoLayersCNN with the specified number of output classes.

        Parameters:
            num_classes (int): The number of classes in the classification task.
        """
        
        super(TwoLayersCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (Tensor): The input tensor containing the batch of images.

        Returns:
            Tensor: The output tensor containing the logits of the network for each class.
        """
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_model(model_name, categories):
    """
    Constructs a neural network model based on the specified model name and number of output classes derived from the categories list.

    Parameters:
        model_name (str): The name of the model to build. Supported models include 'twolayerscnn', various versions of 'resnet',
                          'efficientnet', and 'vit' architectures.
        categories (list): A list of categories which determines the number of output classes.

    Returns:
        nn.Module: The constructed neural network model with the final layer adjusted to match the number of categories.

    Raises:
        ValueError: If the model_name is not supported.
    """
    
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

def train_model(model, device, train_dataset, test_dataset, path_model, batch_size=32):
    """
    Trains a neural network model using specified training and validation datasets with options to save the best model based on accuracy.

    Parameters:
        model (nn.Module): The neural network model to train.
        device (torch.device): The device (GPU or CPU) on which to train the model.
        train_dataset (Dataset): The dataset to use for training.
        test_dataset (Dataset): The dataset to use for validation.
        path_model (str): Path to save the best performing model.
        batch_size (int, optional): Number of samples per batch of computation. Default is 32.

    Returns:
        dict: A dictionary containing metrics such as training and testing loss and accuracy, precision, recall, F1 score,
        and confusion matrices across epochs.
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
    max_accuracy = 0

    epoch = 0
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
        #print("Train Confusion Matrix: ", train_confusion_matrix[-1])
        #print("Test Confusion Matrix: ", test_confusion_matrix[-1])
        
        # Update max accuracy if the current accuracy is greater than its currently value, then dump the current model 
        if (test_accs[-1] > max_accuracy):
            max_accuracy = test_accs[-1]
            torch.save(model, path_model)
            print("Model", path_model, "trained and saved")
        
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
    """
    Sets up a graphical user interface for selecting models and batch sizes for evaluation.

    This function initializes a window with drop-down menus to select a specific model architecture or a batch size for training,
    and a submit button to finalize the selection.
    """
    
    root = tk.Tk()
    root.title = "Model Selection"
    root.geometry("400x300")
    
    # Create checkbox variables
    global CREATE_BATCH_32_MODELS
    global CREATE_BATCH_64_MODELS
    global CREATE_BATCH_128_MODELS
    
    CREATE_BATCH_32_MODELS = tk.BooleanVar(value=True)
    CREATE_BATCH_64_MODELS = tk.BooleanVar(value=True)
    CREATE_BATCH_128_MODELS = tk.BooleanVar(value=True)
    
    # Labels
    model_label = tk.Label(root, text="Select the model(s) to evaluate")

    # Dropdowns
    model_choices = ["resnet", "efficientnet", "vit", "twolayerscnn", "All"]
    model_dropdown = ttk.Combobox(root, values=model_choices)
    
    # Checkboxes
    create_batch_32_models = Checkbutton(root, text="Create Batch 32 Models", variable=CREATE_BATCH_32_MODELS)
    create_batch_64_models = Checkbutton(root, text="Create Batch 64 Models", variable=CREATE_BATCH_64_MODELS)
    create_batch_128_models = Checkbutton(root, text="Create Batch 128 Models", variable=CREATE_BATCH_128_MODELS)
    
    # Set default values
    model_dropdown.set(MODEL_CHOICE)
    create_batch_32_models.select()
    create_batch_64_models.select()
    create_batch_128_models.select()

    # Packing
    model_label.pack(anchor='center', padx=10, pady=5)
    model_dropdown.pack(anchor='center', padx=10, pady=5)
    create_batch_32_models.pack(anchor='center', padx=10, pady=5)
    create_batch_64_models.pack(anchor='center', padx=10, pady=5)
    create_batch_128_models.pack(anchor='center', padx=10, pady=5)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit_click(model_dropdown, root))
    submit_button.pack(anchor='center', padx=10, pady=20)

    # Keep the main loop running until explicitly destroyed
    root.mainloop()

def on_submit_click(model_dropdown, root):
    """
    Handles the event triggered by clicking the submit button in the GUI. This function retrieves the selected model and batch size
    from the dropdowns, updates global variables accordingly, and destroys the GUI root after a slight delay.

    Parameters:
        model_dropdown (ttk.Combobox): Dropdown menu containing the model options.
        batch_size_dropdown (ttk.Combobox): Dropdown menu containing the batch size options.
        root (tk.Tk): The root window of the GUI.
    """
    
    global MODEL_CHOICE, BATCH_SIZE_CHOICE
    
    MODEL_CHOICE = model_dropdown.get()
    
    # Postpone root destruction to after the main loop
    root.after(100, root.destroy)
    
def model_oom_safety_check(model_name, batch_size):
    """
    Checks if a given combination of model name and batch size is known to cause out-of-memory (OOM) issues on the GPU.

    Parameters:
        model_name (str): The name of the model to evaluate.
        batch_size (int): The batch size to evaluate.

    Returns:
        bool: True if the combination is known to be unsafe and may cause OOM errors, False otherwise.
    """
    
    is_model_oom_unsafe = False
    
    # Skip combinations of model_name and batch_size that cause an out of memory exception to throw
    if ((model_name in ["resnet50", "efficientnet_b0", "efficientnet_b1"] and batch_size == 128) or 
        (model_name in ["resnet101", "resnet152", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "vit_b_16", "vit_b_32", "twolayerscnn"] and batch_size in [64, 128])):
        is_model_oom_unsafe = True

    return is_model_oom_unsafe
    
def main():
    """
    The main function to run the model training processes. It initializes the GUI for selection, processes the selections, and
    manages dataset loading and model training based on user inputs. Also includes safety checks for OOM and saves models
    and their histories based on training performance.
    """
    
    setup_gui()
    models_to_run = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet_b0", "efficientnet_b1", "efficientnet_b3", "efficientnet_b4", "vit_b_16", "vit_b_32", "twolayerscnn"]
    batch_sizes = []
    
    if MODEL_CHOICE:
        if MODEL_CHOICE == "resnet":
            models_to_run = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        elif MODEL_CHOICE == "efficientnet":
            models_to_run = [ "efficientnet_b0", "efficientnet_b1", "efficientnet_b3", "efficientnet_b4"]
        elif MODEL_CHOICE == "vit":
            models_to_run = ["vit_b_16", "vit_b_32"]
        elif MODEL_CHOICE == "twolayerscnn":
            models_to_run = ["twolayerscnn"]
            
    if (CREATE_BATCH_32_MODELS.get()):
        batch_sizes.append(32)
    if (CREATE_BATCH_64_MODELS.get()):
        batch_sizes.append(64)
    if (CREATE_BATCH_128_MODELS.get()):
        batch_sizes.append(128)
    
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
                if (model_oom_safety_check(model_name, batch_size)):
                    print(f"Skipping training for {model_name} with batch size {batch_size} to avoid OOM error")
                    continue
                model = build_model(model_name, categories).to(device)
                history = train_model(model, device, train_dataset, test_dataset, path_model, batch_size=batch_size)
                with open(path_history, "wb") as f:
                    pickle.dump(history, f)
            # OOM
            except RuntimeError as e:
                print(str(e))
                print("Model", name, "OOM, skipping")
            except Exception as e:
                print(str(e))
                print("Model", name, "Error", e)
                continue

    histories = []
    for model_name in models_to_run:
        for batch_size in batch_sizes:
            history_file = os.path.join(BASE_DIRECTORY, "Histories", f"{model_name}_{batch_size}.pkl")
            
            if (os.path.isfile(history_file)):
                histories.append(history_file)

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
            train_acc_best="{:.4f}".format(max(history[0]['train_accs'])),
            test_acc_best="{:.4f}".format(max(history[0]['test_accs'])),
        ))

    result_df = pd.DataFrame(data)
    print(result_df.to_markdown())

    _, axes = plt.subplots(math.ceil(len(histories) / 4), 4, figsize=(15, 15))
    axes = axes.flatten()

    for i, hist in enumerate(histories):
        with open(hist, "rb") as f:
            history = pickle.load(f)
        name = op.basename(hist).replace(".pkl", "")
        ax = axes[i]
        plot_history(history[0], ax=ax)
        ax.set_title(name)
        

    plt.tight_layout()
    plt.show()
        
if __name__ == '__main__':
    main()