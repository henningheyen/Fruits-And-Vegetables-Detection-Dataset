# Fruits-And-Vegetables-Detection-Dataset

**TLDR**: This repo contains ...
- the biggest fruits and vegetable YOLO formatted image dataset for object detection with **63 classes** and **8221 images**.
- three YOLOv8 fine-tuned baseline models (`medium`, `large`, `xlarge`).
- sample application demo for scoring the healthiness of meals
- Test it online **[here](https://hub.ultralytics.com/projects/AIhZh0lIAJko7snRmM5f)** 

## The Dataset

<div>
    <img src="Figures/LVIS_Sample_Images/lvis1.jpg" width="150"/>
    <img src="Figures/LVIS_Sample_Images/lvis2.jpg" width="150"/>
    <img src="Figures/LVIS_Sample_Images/lvis3.jpg" width="150"/>
    <img src="Figures/LVIS_Sample_Images/lvis4.jpg" width="150"/>
</div>

<div>
    <img src="Figures/LVIS_Sample_Images/lvis5.jpg" width="150"/>
    <img src="Figures/LVIS_Sample_Images/lvis6.jpg" width="150"/>
    <img src="Figures/LVIS_Sample_Images/lvis7.jpg" width="150"/>
    <img src="Figures/LVIS_Sample_Images/lvis8.jpg" width="150"/>
</div>

- The dataset is a subset of the [LVIS dataset](https://www.lvisdataset.org) which consists of 160k images and 1203 classes for object detection. It is originally COCO-formatted (`.json` based).
- The dataset has been converted from COCO format (`.json`) to YOLO format (`.txt` based)
- All images that do not contain any fruits or images have been removed, resulting in `8221` images and `63` classes (`6721`train, `1500` validation). Additional `180` test images have been manually labelled with Roboflow
- The classes cover most common fruits and vegetables (see [fruits_vegetables_LVIS_dict.json](fruits_vegetables_LVIS_dict.json)
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

# Fine tuning the model on custom dataset
results = model.train(data='LVIS_Fruits_And_Vegetables/data.yaml', epochs=50, imgsz=640)
```

Training locally can be time-consuming. The models have been trained using Ultrylitics HUB Cloud compute resources. Three models have been trained based on different sizes of the pre-trained models (`yolov8m`, `yolov8l`, `yolov8x). The model weights for each model have been stored in `.pt` format in the [Model_Weights](Model_Weights) folder (just unzip). Performance ranges from `0.152` for the `medium` model to `0.202` for the `xlarge` model measured by mAP50-95 (see [Figure](Figures/mAP50-95 by Model Size.png)).

## Inference

The [demo](demp.ipynb) contains examples of how to run inference. Example images with detection using the `xlarge` model can be found in the [Example_Results](Example_Results) folder. 

To test the models on a web interface check out the **[Ultralytics Inference API](https://hub.ultralytics.com/projects/AIhZh0lIAJko7snRmM5f)** under the `Preview` tab. 

<div>
    <img src="Figures/inference_api.png" width="750"/>
</div>

**Example images from test set**: 
<div>
    <img src="Example_Results/20240327_cf3f38d0-7783-4ced-9409-1619c54978f2_2_png.rf.67158dd9f55216861c1d70a108c0f6a6.jpg.png" alt="Example 1" width="150"/>
    <img src="Example_Results/20240328_d2e6cc91-5c10-4f25-9a29-924f7c25a5ad_2_png.rf.33245658e43630ece456a011bd732270.jpg.png" alt="Example 2" width="150"/>
    <img src="Example_Results/20240328_da48c0fd-c904-41ea-a71a-a3b5658b491b_2_png.rf.41265e816ff7564e1844e62f1fc6c470.jpg.png" alt="Example 3" width="150"/>
    <img src="Example_Results/20240403_4432624f-9fac-4d78-afab-742d358eb95c_1_png.rf.713a2208111a7804cbf8635cc4335861.jpg.png" alt="Example 3" width="150"/>
    <img src="Example_Results/20240404_ad61f4d1-5de3-4bc0-9ca1-5739e6c27b93_1_png.rf.7d373ebdd27bfa61c314f9412feed63d.jpg.png" alt="Example 3" width="150"/>
</div>

## Sample Application: Meal Scoring

In the [meal_scoring](meal_scoring.ipynb) demo applies the fine-tuned models to score the healthiness of meals. A use case could be that users upload photos of their meals and with one simple and one complex scoring algorithm, the user will then be rewarded points depending on how colourful and healthy the meal is. 

<div>
    <img src="Figures/meal_scoring_example.png" alt="Example 1" width="500"/>
</div>

## Future Steps

- To achieve better performance the class distribution has to be more uniform. As depicted below most detections in the train set are `Bananas`, `Carrots` and `Apples`.

<div>
    <img src="Figures/Number of Occurrences per Class in the Training Dataset.png" alt="DatasetDistribution" width="800" height="400"/>
</div>



