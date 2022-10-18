Remote sensing image recognition based on FCN

Due to the small dataset, the dataset is first segmented and stored in VOC dataset format.

The image edge extraction algorithm is proposed to use Canny algorithm.

**main.py: Using to test with integrated test code and prediction code. The test section holds 44 sheets of clipped data and labels that can be used to test model performance. The prediction code can be used to predict the labels of 88 cropped images.**

**The results can be obtained by running main() directly; because more datasets cannot be uploaded, only a small number of datasets are used for testing, and the accuracy obtained deviates from MIoU to some extent, but the difference is not large.**



train.py : The file for training the model on the training set, and save the trained model parameters to the pth file under the model folder.

predict.py: call the model trained by the training set, and get the prediction result atlas by pixel prediction of the three images segmented in the test set.

models.py: contains the definition of VGG16 model, FCN-8s model, and migration learning by training with the ImageNet dataset.

voc_loader.py: store the segmented training set and test set images and labels in VOC dataset format.

cut_pic.py: split the training set as well as the test set images in equal proportion. (including labels)

opencv_cut_picture.py: Merge images

tools.py: convert prediction results from network output to images (GitHub lookup)

loss.py: focal loss function implementation, instead of the original cross-entropy loss function, to solve the sample imbalance problem.

edge_detection.py: image edge extraction algorithm for object boundary part for classification to improve the accuracy of image segmentation.

VOC2012: the segmented and organized dataset in VOC dataset format.

result: test set prediction results