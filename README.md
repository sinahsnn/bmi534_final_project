# Context Aware Image Caption Generation Using Deep Learning Algorithms

This repository contains multiple deep learning-based **image captioning** models, exploring various encoder-decoder architectures for generating natural language descriptions of images. It's part of a research project focused on applying **neural networks** to computer vision and NLP tasks.

### Dataset ###
In this study, we used the Flickr8k dataset, a publicly available benchmark dataset specifically designed for image-to-sentence description tasks. This dataset comprises 8,000 images, each annotated with five human-generated captions, resulting in rich textual descriptions capturing diverse events and scenarios. The dataset intentionally excludes images of well-known people and iconic locations, enhancing its general applicability and reducing dataset-specific biases.

This dataset is divided into three subsets: a training set containing 6,000 images, a development (validation) set of 1,000 images, and a test set of 1,000 images.

### Code usage instructions ###
~~~
$ git clone https://github.com/sinahsnn/bmi534_final_project.git
$ cd bmi534_final_project/ 
~~~
## 📁 Repository Structure
- [`adaptive_image_captioning.py`](adaptive_image_captioning.py)
  
- [`image_captioning_efficientb3.py`](image_captioning_efficientb3.py)  
  Image captioning using **EfficientNet-B3** as encoder.

- [`image_captioning_resnet.py`](image_captioning_resnet.py)  
  Uses a **ResNet** backbone to extract visual features for captioning.

- [`image_captioning_unet.py`](image_captioning_unet.py)  
  Explores **U-Net**, typically used for segmentation, as a captioning encoder.

- [`image_captioning_vit.py`](image_captioning_vit.py)  
  Uses **Vision Transformer (ViT)** for image captioning.

- [`resnet34.py`](resnet34.py)  
  Contains ResNet-34 architecture definition used in captioning pipeline.

---
