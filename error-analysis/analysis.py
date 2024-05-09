import requests
import tkinter as tk
import glob
import os
from PIL import Image, ImageTk
from pathlib import Path
import shutil

URL = "http://127.0.0.1:5000/predict"

def get_image_map (path):
    # Structure for images to be inferred:
    # Main Folder : SubFodlers
    # Subfolders -> Picture
    
    folder_paths = []
    for dir in Path(path).iterdir():
        if os.path.isdir(dir):
            folder_paths.append(dir)

    if not folder_paths:
        folder_paths.append(path) # Append the root file if nothing came up.
    
    image_map = []
    tested_categories = []       
    
    for folder in folder_paths:
        if folder:
            image_files = glob.glob(os.path.join(folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(folder, '*.png')))
            
            for image_file in image_files:
                cat_name = (os.path.basename(folder).split('-')[1])
                
                image_map.append([ image_file, cat_name ])
                tested_categories.append(cat_name)
    
    return image_map, list(set(tested_categories))

def construct_confusion_matrix (path):
    image_map, tested_categories = get_image_map(path)

    conf_matrix = []
    
    for record in image_map:
        file = {'file': open(record[0], 'rb')}
        response = requests.post(URL, files=file)
        response_text = response.text.split('"')[3]  # Assuming response.text is suitable for display
        conf_matrix.append([record[0], record[1], response_text]) # Path real_type predicted_type
    
    return conf_matrix, tested_categories

def get_conf_matrix(category, local_result):
    result_true = list(filter(lambda instance: instance[1] == instance[2], local_result))
    result_false = list(filter(lambda instance: instance[1] != instance[2], local_result))

    return f"{category} {len(result_true)} {len(result_false)}", result_false
    
def get_category_instances(category, conf_matrix):
    return list(filter(lambda instance: instance[1] == category, conf_matrix))

def dump (lst, file_path):
    with open(file_path, 'a') as file:
        for item in lst:
            fileName = os.path.basename(item[0])
            copy_file(item[0], "D:\\APML\\results\\" + item[1], fileName)
            file.write(item[0] + "  " + item[1] + "  " + item[2] + '\n')
            
def copy_file(source_path, mainFolder, fileName):
    try:
        if os.path.exists(mainFolder) == False:
            os.mkdir(mainFolder)
        dest_path = mainFolder + "\\" + fileName
        shutil.copy(source_path, dest_path)
       # print(f"File copied from '{source_path}' to '{dest_path}' successfully.")
    except IOError as e:
        print(f"Unable to copy file. {e}")

def print_confusion_matrix(conf_matrix, tested_categories):
    print("-- Confusion Matrix --")
    print("Category Correct Incorrect")
    for category in tested_categories:
        resultStr, false_list = get_conf_matrix (    
                                category, 
                                get_category_instances(category, conf_matrix)
                            )
        print (resultStr)
        dump(false_list, "D:\\APML\\test_folder\\false_tests.txt")
        

def error_analyser(path): # Pass this the path to the folder.
    conf_matrix, tested_categories = construct_confusion_matrix(path)
    print_confusion_matrix(conf_matrix, tested_categories)


print("Starting confusion matrix constructor ..... bulk inference might take some time .... ")
error_analyser("D:\\APML\\train_images")