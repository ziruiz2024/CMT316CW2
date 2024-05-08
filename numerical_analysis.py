

import os
import os.path as op
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
import xml.etree.ElementTree as ET
import pickle
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.notebook import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
import math

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

def numerical_analysis(dftrain, dftest):
    count_category_train = {}
    width_height_list_train = []
    # Count the number of objects per category train set 
    for index, row in dftrain.iterrows():
        width = row['width']
        height = row['height']
        width_height_list_train.append([width, height])
        label = row['label']
        if label in count_category_train:
            count_category_train[label] +=1
        else:
            count_category_train[label] = 1
            
    # Count the number of objects per category test set       
    count_category_test = {}
    width_height_list_test = []
    for index, row in dftest.iterrows():
        width = row['width']
        height = row['height']
        width_height_list_test.append([width, height])
        label = row['label']
        if label in count_category_test:
            count_category_test[label] +=1
        else:
            count_category_test[label] = 1

    # Caculate Difference between highest object count and lowest object count across categories
    highest_count = max(count_category_test.values())
    smallest_count = min(count_category_test.values())
    difference = highest_count - smallest_count
    print(f"Difference between highest object count and lowest object count across categories: {difference}")

    # Caculate standard deviation
    object_counts = list(count_category_test.values())
    total = sum(object_counts)
    mean = total / len(object_counts)
    squared_deviations = [ (x - mean)**2 for x in object_counts]
    sum_of_squared_deviations = sum(squared_deviations)
    n_minus_one = len(object_counts) - 1
    standard_deviation = math.sqrt(sum_of_squared_deviations / n_minus_one)
    print(f"Standard deviation of test obejct counts: {standard_deviation}")
    
    widths_train = [coord[0] for coord in width_height_list_train]
    heights_train = [coord[1] for coord in width_height_list_train]

    # Plot the scatter graph for train set
    plt.figure(figsize=(10, 6))
    plt.scatter(widths_train, heights_train, color='skyblue', marker='o', alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Width vs Height Scatter Plot Train Set')
    plt.grid(True)
    plt.tight_layout()  
    plt.savefig('scatter_graph_train.png')
    plt.show()

    widths_test = [coord[0] for coord in width_height_list_test]
    heights_test = [coord[1] for coord in width_height_list_test]

    # Plot the scatter graph for test set
    plt.figure(figsize=(10, 6))
    plt.scatter(widths_test, heights_test, color='skyblue', marker='o', alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Width vs Height Scatter Plot Test Set')
    plt.grid(True)
    plt.tight_layout() 
    plt.savefig('scatter_graph_test.png')
    plt.show()

    # Extract keys and values
    keys_train = list(count_category_train.keys())
    values_train = list(count_category_train.values())
    
    # Plot category counts
    plt.figure(figsize=(10, 6))
    plt.bar(keys_train, values_train)
    plt.title('Counts for each category train set')
    plt.xlabel('Category')
    plt.ylabel('Number of objects')
    plt.savefig('bar_graph_train.png')
    plt.show()

    # Extract keys and values
    keys_test = list(count_category_test.keys())
    values_test = list(count_category_test.values())
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(keys_test, values_test)
    plt.title('Counts for each category test set')
    plt.xlabel('Category')
    plt.ylabel('Number of objects')
    plt.savefig('bar_graph_test.png')
    plt.show()
        
def main():
    # Read dataset and gather into dataframe
    train_data, test_data, categories = load_dataset()
    dftrain = pd.DataFrame(train_data)
    dftest = pd.DataFrame(test_data)
    numerical_analysis(dftrain, dftest)

if __name__ == '__main__':
    main()