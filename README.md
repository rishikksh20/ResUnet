# Deep ResUnet and ResUnet ++ 
Unofficial Pytorch implementation of following papers :
* [Deep ResUnet](https://arxiv.org/pdf/1711.10684.pdf)
* [ResUnet ++](https://arxiv.org/pdf/1911.07067.pdf)

## Note
* This repo written for experimentation (fun) purpose and heavily hard coded, so avoid to use this as it is in production environement.
* I only wrote ResUnet and ResUnet++ model, Unet is pre-implemented and borrows from this [repo](https://github.com/jeffwen/road_building_extraction).
* Use your own pre-processing and dataloader, dataloader and pre-processing of this repo written for specific use case.
* This repo only tested on [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/).

## Pre-processing
* This pre-processing is for specific use case and follows strict directory structure.
````buildoutcfg
python preprocess.py --config "config/default.yaml" --train training_files_dir --valid validation_files_dir
````
* Training and validation directories passed in `args` above should contain two folders `input` for input images and `output` for target images. And all images are of fixed square size (in this case `1500 * 1500` pixels).
* Pre-processing crop each input and target image into several fixed size (in this case `224 * 224`) small cropped images and saved into `input_crop` and `mask_crop` respectively on training and validation dump directories as in `config` file.
* You can change training and validation dump directories from config file i.e. `configs/default.yaml`.
## Training
```buildoutcfg
python train.py --name "default" --config "config/default.yaml"
```
For Tensorboard:
``tensorboard --logdir logs/
``
## References
- [DenseASPP for Semantic Segmentation in Street Scenes](https://github.com/DeepMotionAIResearch/DenseASPP)
- [ResUNet++ with Conditional Random Field](https://github.com/DebeshJha/ResUNetplusplus_with-CRF-and-TTA)
- [SENet](https://github.com/moskomule/senet.pytorch)
- [Road Extraction Using PyTorch](https://github.com/jeffwen/road_building_extraction)
- [ASPP Module](https://medium.com/@aidanaden/deeplabv3-pytorch-code-explained-line-by-line-sort-of-19e729bb2af6)
- [Deep Residual-Unet](https://arxiv.org/pdf/1711.10684.pdf)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
- [ResUNet++](https://arxiv.org/pdf/1911.07067.pdf)
- [Unet](https://arxiv.org/pdf/1505.04597.pdf)
- [Brain tumor segmentation](https://github.com/galprz/brain-tumor-segmentation)
