
# Semantic segmentation of remote sensing images based on FCN
## Introduction

In **Semantic segmentation of remote sensing images based on FCN** we improve the traditional FCN to achieve the most accurate remote sensing image semantic segmentation task possible. In this paper, we focus on improving the downsampling layer of FCN to solve such a challenge. The improvements are performed using ResNet50 and Swin transformer. and use a reasonable loss function.

This implementation:

- has the demo and training code for this project implemented purely in PyTorch,
- FCNformer base on swin transformer and FCN
- FCNResNet base on ResNet50 and FCN
- FCNVGG base on VGG_16 and FCN
- Focal loss & Dice loss

## Implementation of code

- main.py: The main program that runs the code. Implement training, evaluation and image prediction by tuning the config.
- train.py : The file for training the model on the training set, and save the trained model parameters to the pth file under the model folder.
- predict.py: call the model trained by the training set, and get the prediction result atlas by pixel prediction of the three images segmented in the test set.
- **model**
  - FCNformer.py: Implementation of swin transformer based FCN model
  - fcn_resnet.py: Implementation of ResNet based FCN model
  - models.py: contains the definition of VGG16 model, FCN-8s model, and migration learning by training with the ImageNet dataset.
  - swin.py: Swin transformer implement.
  - resnet.py: resnet model implement.
- **dataloaders**
  - GID_loader.py: store the GID-5 segmented training set and test set images and labels in VOC dataset format.
  - LoveDa_loader.py: store the LoveDa segmented training set and test set images and labels in VOC dataset format.
- **loss**
  - focal.py:  focal loss function implementation, instead of the original cross-entropy loss function, to solve the sample imbalance problem.
  - dice.py:  dice loss function implementation, instead of the original cross-entropy loss function, to solve the sample imbalance problem.
- **Toolkits** (Code to implement the required functions, including evaluation, image cutting, restoration, etc.)
  - edge_detection.py: Edge detection algorithm for image edge extraction.
  - cut_pic.py: Cut the image to a size that can be fed into the network and save it locally.
  - opencv_merge_picture.py: Merge images

## Dataset

Please download The GID-5 dataset in [here](https://x-ytong.github.io/project/GID.html), and save in data filder.

Please download The LoveDA dataset in [here](https://zenodo.org/record/5706578#.Y2Ty5nZBxD8), and save in data filder.

If you want to train this dataset, please check the Toolkits/cut_pic.py file first. Cut the image size to the right size (224*224).

## Running Main for predict

The prediction results of remote sensing images with the completed training model are obtained. The results are saved to the output folder.

```bash
# Please pre-adjust args.predict = 1 in main.py
python main.py

```

## Training

```
# Select the dataset to be trained and the model in the args config snippet.
python main.py

```

## Evaluation

Here we show the tested performance of the three FCN models on the GID-5 and LoveDA datasets with Mean cross-merger ratio (MIoU)

| Models         | GID-5 | LOVEDA |
|:--------------:|:----:|:------------:|
| FCN_VGG16  | 47.32 |     46.69     |
| FCN_ResNet50 | 53.67 |     48.49     |
| FCNformer  | **56.93** |     **49.42**     |

