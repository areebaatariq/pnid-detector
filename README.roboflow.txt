
Object Detection - v7 2025-11-05 5:15pm
==============================

This dataset was exported via roboflow.com on November 5, 2025 at 12:37 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 825 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 10 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random exposure adjustment of between -10 and +10 percent

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* Salt and pepper noise was applied to 0.1 percent of pixels


