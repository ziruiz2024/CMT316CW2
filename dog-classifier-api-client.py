import requests
import tkinter as tk
import glob
import os
from tkinter import filedialog, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk

URL = "http://127.0.0.1:5000/predict"

def browse_image_folder(frame):
    image_paths = []
    path = filedialog.askdirectory(title="Select Folder")
    if path:
        image_files = glob.glob(os.path.join(path, '*.jpg'))
        image_files.extend(glob.glob(os.path.join(path, '*.png')))
        for image_file in image_files:
            image_paths.append(image_file)
        
        display_images(frame, image_paths)

def display_images(frame, image_paths):
    num_images = len(image_paths)
    num_columns = 4
    num_rows = (num_images + num_columns - 1) // num_columns

    # Clear existing images and labels
    for widget in frame.winfo_children():
        widget.destroy()

    # Reset grid configuration
    for i in range(num_rows * 2):  # times 2 because labels are also in the grid
        for j in range(num_columns):
            frame.grid_rowconfigure(i, weight=1)
            frame.grid_columnconfigure(j, weight=1)

    # Place images and labels in the grid
    for index, image_file in enumerate(image_paths):
        img = Image.open(image_file)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # Display the image
        label_image = tk.Label(frame, image=photo)
        label_image.image = photo
        row = index // num_columns * 2  # Multiply by 2 to account for label rows
        column = index % num_columns
        label_image.grid(row=row, column=column, sticky='nsew', padx=5, pady=5)
        
        # Make API request
        file = {'file': open(image_file, 'rb')}
        response = requests.post(URL, files=file)
        response_text = response.text.split('"')[3]  # Assuming response.text is suitable for display
        
        # Display the response label
        label_response = tk.Label(frame, text=response_text)
        label_response.grid(row=row + 1, column=column, sticky='nsew', padx=5, pady=5)


def setup_gui():
    root = tk.Tk()
    root.title("Image Loader")
    root.geometry("900x600")
    
    canvas = Canvas(root)
    scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    frame = scrollable_frame
    
    # Labels
    picture_label = tk.Label(root, text="Select the path to the folder of images you want to classify")
    
    # Create a button that when clicked will open the file dialog
    browse_images_button = tk.Button(root, text="Browse for Images", command=lambda: browse_image_folder(frame))
    browse_images_button.pack(anchor='center', padx=10, pady=5)
    # Packing
    picture_label.pack(anchor='center', padx=10, pady=5)
    
    canvas.pack(side='left', expand=True, fill='both')
    scrollbar.pack(side="right", fill="y")

    # Keep the main loop running until explicitly destroyed
    root.mainloop()

def main():
    setup_gui()

if __name__ == '__main__':
    main()