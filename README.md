# Image Classification of British Currency: MobileNetV2 vs InceptionV3
## What it does
This project classifies images of British banknotes and coins into ten denomination classes, covering 1p, 2p, 5p, 20p, 50p, £1, £5, £10, £20, and £50. It compares two deep learning architectures, MobileNetV2 and InceptionV3, using transfer learning from models pretrained on ImageNet, then fine tunes both on a custom currency dataset.

## Why I built it
This was a coursework assignment for my MSc in Artificial Intelligence at the University of Salford, for the Deep Learning module.

## Tools used
Python, TensorFlow, MobileNetV2, InceptionV3, Roboflow for dataset preparation and augmentation, and testing across both Google Colab and a local Ubuntu Linux environment.

## How to run it
1. Clone this repository and install the dependencies.

2. Place the currency image dataset in the input folder, organised into one subfolder per denomination.

3. Run the preprocessing notebook to clean the dataset, filter out low resolution images, rescale pixel values, and apply data augmentation.

4. Run the training notebook to fine tune MobileNetV2 and InceptionV3 separately on the prepared dataset.

5. Run the evaluation notebook to generate accuracy, precision, F1 score, and confusion matrices for both models.

## Results
MobileNetV2 reached a test accuracy of 62.96 percent, with a precision of 72.10 percent and an F1 score of 64.60 percent, outperforming InceptionV3 overall. InceptionV3 reached a test accuracy of 40.74 percent after additional fine tuning. The confusion matrices showed that both models struggled most with visually similar denominations, particularly separating 1p from 5p coins. MobileNetV2 proved better suited to this task overall due to its efficiency and lower risk of overfitting, while InceptionV3 showed some advantage in recognising higher value notes. The project also compared training on Google Colab against a local Ubuntu machine with GPU support, finding the local setup faster for extended training runs.
