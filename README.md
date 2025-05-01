# Context Aware Image Caption Generation Using Deep Learning Algorithms

This repository contains multiple deep learning-based **image captioning** models, exploring various encoder-decoder architectures for generating natural language descriptions of images. It's part of a research project focused on applying **neural networks** to computer vision and NLP tasks.

### 📦 Dataset ###
In this study, we used the Flickr8k dataset, a publicly available benchmark dataset specifically designed for image-to-sentence description tasks. This dataset comprises 8,000 images, each annotated with five human-generated captions, resulting in rich textual descriptions capturing diverse events and scenarios. The dataset intentionally excludes images of well-known people and iconic locations, enhancing its general applicability and reducing dataset-specific biases.

This dataset is divided into three subsets: a training set containing 6,000 images, a development (validation) set of 1,000 images, and a test set of 1,000 images.

### Code usage instructions ### 
~~~
$ git clone https://github.com/sinahsnn/bmi534_final_project.git
$ cd bmi534_final_project/ 
~~~
## 📁 Repository Structure
- [`adaptive_image_captioning.py`](adaptive_image_captioning.py)  
  Image captioning using **Inception-V3** as the encoder backbone..
  
- [`image_captioning_efficientb3.py`](image_captioning_efficientb3.py)  
  Implements captioning pipeline using **EfficientNet-B3** for feature extraction.

- [`image_captioning_resnet.py`](image_captioning_resnet.py)  
  Uses **ResNet** as a visual encoder for generating image captions..

- [`image_captioning_unet.py`](image_captioning_unet.py)  
  Explores **U-Net**, typically used for segmentation, as a captioning encoder.

- [`image_captioning_vit.py`](image_captioning_vit.py)  
  Leverages the **Vision Transformer (ViT)** for transformer-based image captioning.

- [`resnet34.py`](resnet34.py)  
  Contains the **ResNet-34** architecture definition used in the captioning pipeline.
---
### Results ###
<div align="center">

### 📊 Comparison of Models Using BLEU, METEOR, and ROUGE-L Scores

| **Model**         | **BLEU-1** | **BLEU-4** | **METEOR** | **ROUGE-L** |
|-------------------|------------|------------|------------|-------------|
| ResNet18          | 0.1437     | 0.0104     | 0.1069     | 0.1409      |
| EfficientNetB3    | 0.1366     | 0.0136     | 0.1073     | 0.1397      |
| U-net             | 0.1312     | 0.0095     | 0.1013     | 0.1339      |
| ViT               | 0.1098     | 0.0045     | 0.0763     | 0.1037      |
| **InceptionV3**   | **0.2045** | **0.0322** | **0.1675** | **0.2034**  |

</div>
