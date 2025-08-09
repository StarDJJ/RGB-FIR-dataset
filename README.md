

# PM2.5_Measurement_Network

## RGB-FIR-dataset

<span style="font-size:1.2em;">**Introduction** </span>

This is a multispectral air pollution dataset of significant environmental monitoring value, which comprehensively records air pollution monitoring images of Nantong City and its neighboring areas in Jiangsu Province. The dataset contains both RGB visible images and FIR far-infrared images.



<span style="font-size:1.2em;">**Data Scale and Specifications**</span>

- The test dataset contains 4909 pairs of registered RGB-FIR image pairs
- Image resolution is uniformly set to 1280×720 pixels
- All images have undergone rigorous spatial registration to ensure precise correspondence between RGB and FIR images



<span style="font-size:1.2em;">**Collection Details** </span>

Data collection spans multiple dates, with acquisition periods from 6:00 to 18:00 daily, covering observations under various lighting conditions and weather scenarios throughout the day.



<span style="font-size:1.2em;">**Data Access**</span>

 The complete dataset has been uploaded to the Google Drive platform and is available for download via the provided link.https://drive.google.com/file/d/1XlDmHQ-45iy6F37sk2_3DkqwFq_nAA3q/view?usp=drive_link



<span style="font-weight:bold; font-size:1.2em;">Agreement</span>

In order to help us retain the appropriate rights and to regulate your use, we have created a dataset usage agreement.

If you want to use these dataset, please do contact us.

x2507537309@163.com    Starr Bruce



## Preparation

<span style="font-weight:bold; font-size:1.2em;">Install</span>

We use the following environment:

```
torch==1.12.0+cu113
torchvision==0.13.0+cu113
numpy==1.21.2
pandas==1.3.5
opencv_python==4.10.0.84
```

1. Create a new conda environment

```
conda create -n pytorch-gpu python=3.7
conda activate pytorch-gpu
```

2. Install dependencies

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```



<span style="font-weight:bold; font-size:1.2em;">Download</span>

You can download the pretrained models on https://drive.google.com/file/d/1qBduY1PxAekMReoYVa_tUrvtAn4fTKY9/view?usp=drive_link



## Training and Test

<span style="font-weight:bold; font-size:1.2em;">Train</span>

You can modify the training settings for each experiment in the train.py

Training step：

1. Place the dataset in the dataset folder under the root directory.

2. run train.py.

   

<span style="font-weight:bold; font-size:1.2em;">Test</span>

1. Download the pretrained weight file .pth .

2. In the siamese_params.py file, modify model_path so that it corresponds to the trained file; model_path corresponds to the weight file under the weights folder. 

3. run predict.py, input:

```python
../datasets/similarity/southwest_rgb_Moderate_20231208_09_011.jpg
```

```python
../datasets/similarity/southwest_rgb_Moderate_20231208_19_001.jpg
```

