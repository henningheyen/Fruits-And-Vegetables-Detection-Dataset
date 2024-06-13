# Fruits-And-Vegetables-Detection-Dataset

**TLDR**: This repo contains ...
- the biggest fruits and vegetable image YOLO formatted dataset for object detection with 62 classes and 8221 images.
- three YOLOv8 fine tuned model fine-tuned models (`medium`, `large`, `xlarge`).
- sample application demo for scoring the healthiness of meals

## The Dataset

- The dataset is a subset of the [LVIS dataset](https://www.lvisdataset.org) which consists of 160k images and 1203 classes for object detection. It is originally COCO-formatted (`json` based).
- We have converted this dataset to YOLO format (`txt` based)
- All images that do not contain any fruits or images have been removed, resulting in `8221` images and `62` classes (`6721`train, `1500` validation). Additional `180` test images have been manually labeled with Roboflow
- The `LVIS-Fruits-And-Vegetables-Dataset` has also been uploaded to
    - [Kaggle](https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset) and
    - [Ultralytics Dataset HUB](https://hub.ultralytics.com/datasets/F2NY9oa4vTCPgy9chAe5)

## The Model

Getting started:

```
pip install -r requirements.txt
```

To fine-tune a YOLOv8 model the following code can be used as also shown in the [demo](demo.ipynb):

```
from ultralytics import YOLO

# Load a pretrained model (e.g. YOLOv8m)
model = YOLO('yolov8m.yaml')  # build a new model from YAML

# Fine tuning the model on our custom dataset
results = model.train(data='LVIS_Fruits_And_Vegetables/data.yaml', epochs=50, imgsz=640)
```

Training locally can be time-consuming. We have trained the models using Ultrylitics HUB Cloud compute resources. Three models have been trained based on different sizes of the pre-trained models (`yolov8m`, `yolov8l`, `yolov8x). The model weights for each model have been stored in `.pt` format in the [Model_Weights](Model_Weights) folder (just unzip). Performance ranges from `0.152` for the `medium` model to `0.202` for the `xlarge` model measured by mAP50-95 (see [Figure](Figures/mAP50-95 by Model Size.png)).

## Inference

The [demo](demp.ipynb) contains examples on how to run inference. Example images with detection using the `xlarge` model can be found in the [Example_Results](Example_Results) folder. 

To test our models on a web interface check out the **[Ultralytics Inference API](https://hub.ultralytics.com/projects/AIhZh0lIAJko7snRmM5f)** under the `Preview` tab. 

## Future Steps

- To achieve better performance the class distribution has to be more uniform. As depicted below most detections in the train set are `Bananas`, `Carrots` and `Apples`.

<div>
    <img src="Figures/Number of Occurrences per Class in the Training Dataset (LVIS_Mirror).png" alt="DatasetDistribution" width="200"/>
</div>


This repository contains most of the code used for the *Eat The Rainbow* one-week challenge. The goal was to develop a model that detects fruits and vegetables in user-uploaded images and scores them according to a scoring logic (see [Assignment](Presentation_and_Assignment/Home-Assignment-Eat-the-Rainbow.pdf)). 

The [Presentation](Presentation_and_Assignment/Presentation-Mirror_compressed.pdf) summarizes the assignment.

# Examples

<div>
    <img src="Example_Results/result20240327_3c8f4813-06ae-4d71-8dd3-d6f5d2e41c9a_4_png.rf.ef6e00bb24b03c891f248a60686d94f1.jpg.png" alt="Example 1" width="200"/>
    <img src="Example_Results/result20240403_4432624f-9fac-4d78-afab-742d358eb95c_1_png.rf.713a2208111a7804cbf8635cc4335861.jpg.png" alt="Example 2" width="200"/>
    <img src="Example_Results/result20240404_ad61f4d1-5de3-4bc0-9ca1-5739e6c27b93_1_png.rf.7d373ebdd27bfa61c314f9412feed63d.jpg.png" alt="Example 3" width="200"/>
    <img src="Example_Results/result20240327_cf3f38d0-7783-4ced-9409-1619c54978f2_2_png.rf.67158dd9f55216861c1d70a108c0f6a6.jpg.png" alt="Example 3" width="200"/>
    <img src="Example_Results/result20240328_da48c0fd-c904-41ea-a71a-a3b5658b491b_2_png.rf.41265e816ff7564e1844e62f1fc6c470.jpg.png" alt="Example 3" width="200"/>
</div>
