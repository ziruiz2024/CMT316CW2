"""
Image Classification Model Training and Evaluation Tool

This script offers functionality for observing, downloading, and evaluating various machine learning models specialized for image classification tasks. 

Key Features:
- Load image datasets with XML annotations.
- Define and utilize custom dataset classes for handling image transformations.
- Plot training and validation metrics using Plotly for visual comparison.
- Download pretrained model states from remote repositories.
- GUI for selecting models and configurations for training or evaluation.

Dependencies:
- xml.etree.ElementTree: For XML parsing.
- torch: For constructing and training neural network models.
- tkinter: For creating the GUI.
- requests, gdown: For downloading files.
- plotly: For plotting training/validation metrics.
- pandas: For data manipulation.
- os, json, pickle: For file and data handling.
- PIL: For image manipulation.

Usage:
    Run the script directly from the command line to launch the GUI and start model training or evaluation processes based on user inputs.
"""
import xml.etree.ElementTree as ET
import torch
import tkinter as tk
import requests
import plotly.graph_objects as go
import pickle
import pandas as pd
import os
import os.path as op
import json
import gdown
from tqdm import tqdm
from torch.utils.data import Dataset
from tkinter import Checkbutton, ttk
from scipy.io import loadmat
from PIL import Image

# Declare global variables
MODEL_CHOICE = "All"
BATCH_SIZE_CHOICE = "32,64,128"
BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def load_dataset():
    """
    Loads and processes image data from .mat files and XML annotations to prepare training and testing datasets.

    Returns:
        tuple: A tuple containing lists of dictionaries for training and testing data, and a dictionary of category IDs.
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
    """
    A PyTorch dataset class for handling image datasets with customizable transformations.

    Attributes:
        df (DataFrame): DataFrame containing the paths to images and their corresponding labels.
        transform (callable, optional): Optional transform to be applied on a PIL image.
    """
    
    def __init__(self, df, transform=None):
        """
        Initializes the dataset with a DataFrame and an optional transform.

        Parameters:
            df (DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Transform to apply to each image.
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
        Retrieves an image and its label from the dataset at a specified index.
        
        Parameters:
            idx (int): Index of the item.
        
        Returns:
            tuple: Tuple containing the transformed image and its label as a tensor.
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

def get_metric_max_values(metric_data):
    """
    Calculates the maximum values of specified metrics from given data.

    Parameters:
        metric_data (list): List of dictionaries containing model metrics data.
    
    Returns:
        list: List of tuples containing the maximum values and corresponding model names.
    """
    max_values = []
    
    # Collect all maximum values to determine the global maximum
    for model_data in metric_data:
        max_value = max(model_data['values'])
        max_values.append((max_value, model_data['model']))
    
    return max_values
    
def plot_metric_comparison(metrics_data, metric_name):
    """
    Plots comparison graphs for specified metrics across different models using Plotly.

    Parameters:
        metrics_data (dict): Dictionary containing training and testing metrics data.
        metric_name (str): The name of the metric to plot.
    """
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
    """
    Loads and structures training histories from specified files into a usable format.

    Parameters:
        histories_directory (str): Directory where history files are stored.
        models_to_run (list): List of model names to load histories for.
        batch_sizes (list): List of batch sizes for which histories need to be loaded.
    
    Returns:
        dict: Dictionary containing structured training histories.
    """
    all_histories = {}
    
    for model_name in models_to_run:
        for batch_size in batch_sizes:
            history_file = os.path.join(histories_directory, f"{model_name}_{batch_size}.pkl")
            
            if (os.path.isfile(history_file)):
                model_name = history_file.replace(".pkl", "")
                with open(os.path.join(histories_directory, history_file), "rb") as f:
                    history = pickle.load(f)
                for key in history[0].keys():
                    if key not in all_histories:
                        all_histories[key] = []
                    all_histories[key].append({'model': model_name, 'values': history[0][key]})
                    
    return all_histories

def download_pre_built_models(histories_directory, models_to_run, batch_sizes):
    """
    Downloads pre-built model states from specified URLs if they do not already exist locally.

    Parameters:
        histories_directory (str): Directory to save the downloaded model states.
        models_to_run (list): List of model names for which to download states.
        batch_sizes (list): List of batch sizes corresponding to the model states.
    """
    pre_built_model_links = json.loads(json.dumps({"efficientnet_b0_32.pkl":"https://drive.google.com/uc?export=download&id=11eDWSD72mpWP6XgkQ0SY4QVG4Rx59y9V",
                                                    "efficientnet_b0_64.pkl":"https://drive.google.com/uc?export=download&id=14q_-Sj2rTohsI4RTA3cPkzimzBJJk9TO",
                                                    "efficientnet_b1_32.pkl":"https://drive.google.com/uc?export=download&id=1frEXIHsG7kTfhu-hsY93gnwYdD09-03C",
                                                    "efficientnet_b4_32.pkl":"https://drive.google.com/uc?export=download&id=1BbUA0r1GwF5DQj2lE-U1VzEt_dgNcUPf",
                                                    "resnet18_32.pkl":"https://drive.google.com/uc?export=download&id=1QULFR3v3s0jiVuvkwc6WLazA32CGcBT3",
                                                    "resnet18_64.pkl":"https://drive.google.com/uc?export=download&id=145NNvFdIrve1_Sz57jcUX4EChy2lMkxv",
                                                    "resnet18_128.pkl":"https://drive.google.com/uc?export=download&id=1QM6xceookZkUYLdqyJ5aBqYBlbZOTYRW",
                                                    "resnet34_32.pkl":"https://drive.google.com/uc?export=download&id=1-mEg8giPUfHcdVr8luRq0kHR4Oh82uIM",
                                                    "resnet34_64.pkl":"https://drive.google.com/uc?export=download&id=1uEQKdZMnI-DI0Go_1QI8Fz7GinFjZjMY",
                                                    "resnet34_128.pkl":"https://drive.google.com/uc?export=download&id=17GFPN2PbWwMGVHi3jjrMFkglU7LEA5tx",
                                                    "resnet50_32.pkl":"https://drive.google.com/uc?export=download&id=1cJmjxzAMCi0A0VXk6MTWXTvvBuGmUDbH",
                                                    "resnet50_64.pkl":"https://drive.google.com/uc?export=download&id=1LvOczQzBXo1JCQFBiPH2JxJ6PAj_YnzO",
                                                    "resnet101_32.pkl":"https://drive.google.com/uc?export=download&id=1lUglK6oA1wR1SjbNfFGSdRRPm0w54E3T",
                                                    "resnet152_32.pkl":"https://drive.google.com/uc?export=download&id=1VjcARxHKnckpIlgBuXQCtGqLTeF16ZBQ"}))
    
    for model_to_run in models_to_run:
        for batch_size in batch_sizes:
            history_filename = f"{model_to_run}_{str(batch_size)}.pkl"
            history_file_path = os.path.join(histories_directory, history_filename)
            
            if (os.path.isfile(history_file_path)):
                print(f"{history_file_path} already exists, no need to download")
            else:
                # Create a session to handle cookies
                session = requests.Session()
                
                if (history_filename in pre_built_model_links):
                    pre_built_modeld_link = pre_built_model_links[history_filename]
                    print(f"{history_file_path} was not found , it will be downloaded from: {pre_built_modeld_link}.")
                    
                    gdown.download(pre_built_modeld_link, history_file_path)
                else:
                        print(f"{history_file_path} was not found , and a pre built downloaded could not be found for it.")
    
    
def setup_gui():
    """
    Sets up a graphical user interface for selecting models and batch sizes for training or evaluation.
    """
    root = tk.Tk()
    root.title = "Model Selection"
    root.geometry("600x400")
    
    # Create checkbox variables
    global DOWNLOAD_PRE_BUILT_MODELS
    global SHOW_ACCURACY_METRICS
    global SHOW_PRECISION_METRICS
    global SHOW_RECALL_METRICS
    global SHOW_F1_METRICS
    
    DOWNLOAD_PRE_BUILT_MODELS = tk.BooleanVar(value=True)
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
    download_pre_built_models = Checkbutton(root, text="Download Missing Pre Built Models if Available", variable=DOWNLOAD_PRE_BUILT_MODELS)
    show_accuracy_metrics_checkbox = Checkbutton(root, text="Display Accuracy Metrics", variable=SHOW_ACCURACY_METRICS)
    show_precision_metrics_checkbox = Checkbutton(root, text="Display Precision Metrics", variable=SHOW_PRECISION_METRICS)
    show_recall_metrics_checkbox = Checkbutton(root, text="Display Recall Metrics", variable=SHOW_RECALL_METRICS)
    show_f1_metrics_checkbox = Checkbutton(root, text="Display F1 Score Metrics", variable=SHOW_F1_METRICS)
    
    # Set default values
    model_dropdown.set(MODEL_CHOICE)
    batch_size_dropdown.set(BATCH_SIZE_CHOICE)
    download_pre_built_models.select()
    show_accuracy_metrics_checkbox.select()
    show_precision_metrics_checkbox.select()
    show_recall_metrics_checkbox.select()
    show_f1_metrics_checkbox.select()

    # Packing
    model_label.pack(anchor='center', padx=10, pady=5)
    model_dropdown.pack(anchor='center', padx=10, pady=5)
    batch_choices_label.pack(anchor='center', padx=10, pady=5)
    batch_size_dropdown.pack(anchor='center', padx=10, pady=5)
    download_pre_built_models.pack(anchor='center', padx=10, pady=5)
    show_accuracy_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_precision_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_recall_metrics_checkbox.pack(anchor='center', padx=10, pady=5)
    show_f1_metrics_checkbox.pack(anchor='center', padx=10, pady=5)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit_click(model_dropdown, batch_size_dropdown, root))
    submit_button.pack(anchor='center', padx=10, pady=20)

    # Keep the main loop running until explicitly destroyed
    root.mainloop()

def on_submit_click(model_dropdown, batch_size_dropdown, root):
    """
    Handles the event when the submit button is clicked in the GUI. Updates global settings based on user selection.

    Parameters:
        model_dropdown (ttk.Combobox): Dropdown menu for selecting the model.
        batch_size_dropdown (ttk.Combobox): Dropdown menu for selecting the batch size.
        root (tk.Tk): Root window of the GUI.
    """
    global MODEL_CHOICE, BATCH_SIZE_CHOICE
    
    MODEL_CHOICE = model_dropdown.get()
    BATCH_SIZE_CHOICE = batch_size_dropdown.get()
    
    # Postpone root destruction to after the main loop
    root.after(100, root.destroy)
    

def main():
    """
    Main function to run the setup GUI and perform subsequent model training or evaluation based on the GUI inputs.
    """
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
    
    # Read dataset and gather into dataframe
    train_data, test_data, categories = load_dataset()
    dftrain = pd.DataFrame(train_data)
    dftest = pd.DataFrame(test_data)

    # Get the classes summary
    print("Number of classes: ", len(categories))
    print("Number of training samples: ", len(dftrain))
    print("Number of testing samples: ", len(dftest))
    
    if (DOWNLOAD_PRE_BUILT_MODELS):
        download_pre_built_models(os.path.join(BASE_DIRECTORY, "Histories"), models_to_run, batch_sizes)
    
    histories = []
    for model_name in models_to_run:
        for batch_size in batch_sizes:
            history_file = os.path.join(BASE_DIRECTORY, "Histories", f"{model_name}_{batch_size}.pkl")
            
            if (os.path.isfile(history_file)):
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
                train_acc_best="{:.4f}".format(max(history[0]['train_accs'])),
                test_acc_best="{:.4f}".format(max(history[0]['test_accs'])),
            ))
            
        all_histories = load_and_structure_histories(os.path.join(BASE_DIRECTORY, "Histories"), models_to_run, batch_sizes)
        metrics = []
        
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