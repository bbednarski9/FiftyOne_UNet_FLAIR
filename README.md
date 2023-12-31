# FiftyOne - UNet - FLAIR: Deep Learing Tumor Segmentation
## Follow <a href="https://github.com/bbednarski9/FiftyOne_UNet_FLAIR/blob/main/demo_notebook.ipynb">demo_notebook.ipynb</a> for step-by-step AI modeling for brain lesion segmentation using FiftyOne, PyTorch, and a UNet model!

<br>
<strong>Create a 2D-UNet deep learning model in Pytorch to segment brain tumors using the exciting FiftyOne computer vision toolkit to improve data processing and model optimization. UNet tumor segmentations are impressive after <strong>only</strong> only 25 epochs. Inference masks: dark shading, dice loss=0.82; Ground truth masks: white shading</strong>
<br>
<img src="./demo_images/tumor_segment_example3-1.png" alt="">
<br>

## Motivation

Medical imaging is a huge application space for computer vision and AI. The U.S. medical industry is over 17% of national GDP. With the average age of cancer screenings moving up, it's important to facilitate the rapid development of medical CV by allowing practitioners to robustly curate datasets, annotations, and develop models directly on top of the data platform. FiftyOne is positioned to become an irreplacable tool in the model development pipeline. However, medical imaging also offers a unique set of challenges such as unique file types, 3D volumetric data, and unique image metadata per application. Additionally, medical imaging tasks are frequently comprised of "non-natural" images, which may not facilitate acceptable performance from pretrained foundational imaging models.<br>

In this notebook, I demo'd using FiftyOne as a core tool in developing a brain tumor segmentation model from mri. I worked around incompatibilites encountered with the .tif database, visualize the dataset in FiftyOne to inform the development of the subsequent Torch dataloader, and visually confirmed improving segmentation performance of my model in FiftyOne.<br>

## Methods
Models: 2D UNet implemented in PyTorch<br>
Model Weights: available on <a href="https://drive.google.com/drive/folders/1ECU41KCEGHFGLWS6wfiJH6dKN03fDmwV?usp=sharing">This Google Drive</a><br>
Dataset: FLAIR Brain MRI - Segmenting Low-Grade Glioma Tumors<br>
Core Tools: FiftyOne, Pytorch, Numpy<br>

## Results
UNet model achieves Dice Loss of 0.82 after only 25 epochs of training on 90% of FLAIR dataset. 10% witheld for validation. No external or temporal hold-out in this demo.

<br>
<strong>Comparing many samples at once with FiftyOne displays multiple slices in an imaging series by patient</strong>
<br>
<img src="./demo_images/tumor_segment_example3-2.png" alt="">
<br>

## To do:
<ul>
    <li>Find or write a FiftyOne extension to improve viewing of 3D volumes of image slices</li>
    <li>Compare performance to Segment Anything Model - default vith weights were having some issues to look into</li>
    <li>Contribute TotalSegmentator to the FiftyOne model zoo</li>
</ul>


<i>
Prepared by: Bryan Bednarski, September 2023<br>
Github: https://github.com/bbednarski9/FiftyOne_UNet_FLAIR<br>
Email: bryanbed@g.ucla.edu<br>
</i>

