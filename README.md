# DELTAS Research @ Magic Leap (ECCV 2020)
Inference Code for DELTAS: Depth Estimation by Learning Triangulation And densification of Sparse points (ECCV 2020)

## Introduction
DELTAS is a ECCV 2020 research project done at Magic Leap. DELTAS is short for depth estimation by learning triangulation and densification of sparse points. This repo includes PyTorch code and pretrained weights for running the DELTAS network on ScanNet dataset. 

DELTAS operates on a set of posed images to output the metric depth using an end-to-end architecture. For more details, please see:

* Full paper PDF: [DELTAS: Depth Estimation by Learning Triangulation And densification of Sparse points](https://arxiv.org/abs/2003.08933).

* Authors: *Ayan Sinha, Zak Murez, James Bartolozzi, Vijay Badrinarayanan, Andrew Rabinovich*

We provide the pre-trained weights file for the model trained on ScanNet data. Download the pre-trained weights from this [link](https://drive.google.com/uc?export=download&id=1lWjjl44o81m1_lZ2e9CW99OjQhZdd9gN) and place it in the [assets folder](./assets).   

## Dependencies
* Python 3 >= 3.5
* PyTorch >= 1.3.1
* Torchvision >= 0.4.2
* OpenCV >= 3.4 
* NumPy >= 1.18
* Path >= 15.0.0

Simply run the following command: `pip install numpy opencv-python path torch torchvision`. You need a modern GPU to run the inference code in a reasonable time. The code supports running the network without a GPU, albeit with slight degradation in performance. The pre-trained model weights are to be downloaded from [here](https://drive.google.com/uc?export=download&id=1lWjjl44o81m1_lZ2e9CW99OjQhZdd9gN) and to be placed in the [assets folder](./assets).

### Run the network on sample ScanNet data

Run the pre-trained network on sample ScanNet scans. The [sample_data](./assets/sample_data) folder contains 3 frames from a scene in [ScanNet](https://github.com/ScanNet/ScanNet). To test the network on the sample data, simply run:

```sh
python test_learnabledepth.py
```
You should get the following output (or something very close to it)

```txt
=> fetching scenes in './assets/sample_data/scannet_sample'
3 samples found in 1 valid scenes
=> creating model
 TEST: Depth Error 0.1389 (0.0747)
```
Here 0.1389 is the mean absolute error (MAE) in meters and 0.0747 is the relative error. 

### Test the network on ScanNet data

Download and extract Scannet test data (100 scenes) by following the instructions provided at http://www.scan-net.org/.
The directory structure should look like:
```
DATAROOT
└───scannet
│   └───scans_test
│   |   └───scene0000_00
│   |       └───color
│   |       │   │   0.jpg
│   |       │   │   1.jpg
│   |       │   │   ...
│   |       │   ...
```
To test the network on the ScanNet test dataset, run:

```sh
python test_learnabledepth.py --data [DATAROOT] --seq_length 3 --seq_gap 20
```

* The `--data` flag should point to DATAROOT
* The `--seq_length` flag should indicate the number of frames (default:3)
* The `--seq_gap` flag should indicate the stride between frames (default:20)

The arguments in the python file have an associated description. 

## Citation

```
@inproceedings{murez2020atlas,
  title={DELTAS: Depth Estimation by Learning Triangulation And densification of Sparse points},
  author={Ayan Sinha and
  	 Zak Murez and 
  	 James Bartolozzi and
  	 Vijay Badrinarayanan and
  	 Andrew Rabinovich},
  booktitle = {ECCV},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.08933}
}
```

## Additional Notes
* We recommend to run DELTAS at 320x240 resolution. Alternate resolutions will require re-training the network with appropriate data augmentation
* We do not intend to release the DELTAS training code.
* We recommend swapping the native pytorch SVD with this [SVD](https://github.com/KinglittleQ/torch-batch-svd) implementation to fasten training/inference. 


## Legal Disclaimer
Magic Leap is proud to provide its latest samples, toolkits, and research projects on Github to foster development and gather feedback from the spatial computing community. Use of the resources within this repo is subject to (a) the license(s) included herein, or (b) if no license is included, Magic Leap's [Developer Agreement](https://id.magicleap.com/terms/developer), which is available on our [Developer Portal](https://developer.magicleap.com/).
If you need more, just ask on the [forums](https://forum.magicleap.com/hc/en-us/community/topics)!
We're thrilled to be part of a well-meaning, friendly and welcoming community of millions.

