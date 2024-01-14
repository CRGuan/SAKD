## Segmentation - Pascal VOC

### Additional requirements

- tqdm
- matplotlib 
- pillow

### Settings

|   Teacher  |  Student  | Teacher size | Student size | Size ratio |
|:----------:|:---------:|:------------:|:------------:|:----------:|
| ResNet 101 | ResNet 18 |    59.3M    |    16.6    |   28.0%   |
| ResNet 101 | MobileNetV2 |    59.3M    |     5.8M    |   9.8%   |


### Teacher models
Download following pre-trained teacher network and put it into ```./Segmentation/pretrained``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/open?id=1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR)

We used pre-trained model in [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) for teacher network.

### Training

- First, move to segmentation folder : ```cd Segmentation```

- Next, configure your dataset path on ```Segmentation/mypath.py```

- Distillation
  - ResNet 18
  ```shell script
  CUDA_VISIBLE_DEVICES=0,1 python train_with_distillation.py --backbone resnet18 --gpu-ids 0,1 --dataset pascal --use-sbd --nesterov
  ```
  
  -MobileNetV2
  ```shell script
  CUDA_VISIBLE_DEVICES=0,1 python train_with_distillation.py --backbone mobilenet --gpu-ids 0,1 --dataset pascal --use-sbd --nesterov
  ```

### Experimental results

This numbers are based validation performance of our code.

- ResNet 18

|   Network  |  Method  | mIOU |
|:----------:|:--------:|:----------:|
| ResNet 101 |  Teacher |   77.39   |
|  | Baseline |   71.79   |
|  |    KD    |   73.09   |
| ResNet 18 |    AT    |   72.61   |
|  | OFD | 73.24 |
|  | Proposed | __74.02__ |

- MobileNetV2

|  Network  |  Method  |  mIOU |
|:---------:|:--------:|:-----:|
| ResNet 101 |  Teacher | 77.39 |
|  | Baseline | 68.44 |
|  |    KD    |   71.49   |
| MobileNetV2 |    AT    |   71.39   |
|  | OFD | 71.36 |
|  | Proposed | __71.94__ |

### Acknowledgement

The work is based on [OFD](https://github.com/clovaai/overhaul-distillation)(ICCV 2019) , [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception).
