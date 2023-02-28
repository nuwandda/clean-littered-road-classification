# Clean Littered Road Classification
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## Introduction
Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. Typically, Image Classification refers to images in which only one object appears and is analyzed. In contrast, object detection involves both classification and localization tasks, and is used to analyze more realistic cases in which multiple objects may exist in an image.
In this project, we classify roads if clean or littered.

<img src="https://github.com/nuwandda/Clean-Littered-Road-Classification/blob/main/images/dirty.png" width="500" height="400">
<img src="https://github.com/nuwandda/Clean-Littered-Road-Classification/blob/main/images/clean.png" width="500" height="400">

<!-- ARCHITECTURE -->
## Architecture
### The EfficientNetB0 Model
The EfficientNetB0 is the smallest model in the EfficientNet family. With 1000 outputs classes (for ImageNet) in the final fully-connected layer, it has only 5.3 million parameters. This gives around 77.1% top-1 accuracy on the ImageNet dataset. Still, this beats ResNet50 which has 76.0% top-1 accuracy but with 26 million parameters. 
<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- DATASET -->
## Dataset
The dataset contains images of clean and dirty road.

There are a total of 237 images, all of which bootstraped from the internet. The task is to create a classification model, which can accurately classify if a road is clean or littered. Because of the lack of data, pretrained models and data augmentation may be used.

Such a classification model can be used to develope applications to detect littered part of roads using cameras and send necessary service to those areas.
Naviagate Dataset:

    Images: Folder containing all the road images.
    metadata.csv: A csv file mapping the image name with the class label.

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
Instructions on setting up your project locally.
To get a local copy up and running follow these simple steps.

### Install dependencies
To install the required packages. In a terminal, type:
  ```sh
  pip install -r src/requirements.txt
  ```

### Download dataset
To download the dataset, run **prepare_data.py**. You need to create an API key for Kaggle. 
  ```sh
  python prepare_data.py --username username --key api_key
  ```
It will be downloaded inside to **data** folder.

### Training

  ```sh
  python train.py --epochs 50 --learning-rate 0.001 --pretrained
  ```

### Inference
In order to test the trained model, we will use the image in the **test_images** folder. To test the trained model:
  ```sh
  python inference.py
  ```

Colab version will be added.

<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- RESULTS -->
## Results
Download the resulting model [here][1].

You can see the results for the trained model.

Loss:

<img src="https://github.com/nuwandda/Clean-Littered-Road-Classification/blob/main/outputs/loss_pretrained_True.png" width="500" height="400">

Accuracy:

<img src="https://github.com/nuwandda/Clean-Littered-Road-Classification/blob/main/outputs/accuracy_pretrained_True.png" width="500" height="400">

[1]: https://drive.google.com/file/d/1wdNtJza-zInFG4iNFIN07CePNRBULOXd/view?usp=sharing