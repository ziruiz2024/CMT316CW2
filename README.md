# CV-2: Fine-grained image classification (Dogs) Repository

### 1. Download the Stanford Dogs images data from http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar.
###    Extract the .tar file and copy the images folder to the root directory of this project. 
### 2. Download the Stanford Dogs annotations data from http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar.
###    Extract the .tar file and copy the annotation folder to the root directory of this project. 
### 3. Download the Stanford Dogs lists data from http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar .
###    Extract the .tar file and copy the lists folder to the root directory of this project. 
### 4. Run model_builder.py to select and build models via the script’s user interface. 
### 5. Run display_model_metrics.py to evaluate performance metrics of models. You can choose models you've built with 
###    model_builder.py or download pre-built models through the user interface. 
### 6. Run dog_classifier_api_server.py to launch a Flask server that uses selected models. The server accepts image 
###    uploads via POST requests. Models can be self-built or downloaded via the user interface.
### 7. Run dog_classifier_api_client.py to query the dog_classifier_api_server with your images for classification, 
###    managed through the script’s user interface.
 
To run the code, ensure the working directory is in the following structure

```
{working directory}/
    Annotation
    Images
    lists/
        test_list.mat
        train_list.mat
        file_list.mat        
    main.ipynb
```


### 8. History can be found at this link: [Google Drive Document](https://drive.google.com/drive/folders/1_nb9dnsYZfa3Ekt9V1GHcsZ8x-Y5syIL?usp=sharing)
