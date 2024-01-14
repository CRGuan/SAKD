## ImageNet-1K

### Settings

| Setup | Compression type |  Teacher  |  Student  | Teacher size | Student size | Size ratio |
| :---: | :--------------: | :-------: | :-------: | :----------: | :----------: | :--------: |
|  (a)  |      Depth       | ResNet 34 | ResNet 18 |   21797672   |   11689512   |   53.63%   |


In case of ImageNet-1K, teacher model will be automatically downloaded from PyTorch sites.

### Training

- (a) : ResNet 34 to ResNet 18
```
python SAKD_ImageNet1k.py \
--data_path your/path/to/ImageNet \
--net_type resnet \
--epochs 100 \
--lr 0.1 \
--batch_size 256
```



### Experimental results

- ResNet 34 -> ResNet 18

|       | Teacher | Vanilla |  KD   |  AT   |  RKD  |  CRD  |  AFD  |   SRRL    | ReviewKD |  MGD  | MasKD | SAKD(Ours) |
| :---- | :-----: | :-----: | :---: | :---: | :---: | :---: | :---: | :-------: | :------: | :---: | :---: | :--------: |
| Top-1 |  73.31  |  69.75  | 70.66 | 70.59 | 71.34 | 71.17 | 71.38 | **71.73** |  71.61   | 71.58 | 71.26 |   71.67    |
| Top-5 |  91.42  |  89.07  | 89.73 | 90.37 | 90.13 | 90.11 |  N/A  |   90.60   |  90.51   | 90.35 |  N/A  | **90.64**  |



### Requirements

- Python 
- PyTorch



### Acknowledgement

The work is based on [OFD](https://github.com/clovaai/overhaul-distillation)(ICCV 2019) / [RepDistiller](https://github.com/HobbitLong/RepDistiller)(ICLR 2020)

