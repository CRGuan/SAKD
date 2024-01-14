## CIFAR-100

### Settings
We provide the code of the experimental settings specified in the paper.

| Setup | Architecture type |  Teacher   |  Student  | Teacher size | Student size | Size ratio |
| :---: | :---------------: | :--------: | :-------: | :----------: | :----------: | :--------: |
|  (a)  |       Same        | ResNet 56  | ResNet 20 |    0.86 M    |    0.28 M    |   32.56%   |
|  (b)  |       Same        | ResNet 56  | ResNet 32 |    0.86 M    |    0.47 M    |   54.65%   |
|  (c)  |       Same        | ResNet 110 | ResNet 20 |    1.74 M    |    0.28 M    |   16.09%   |
|  (d)  |       Same        | ResNet 110 | ResNet 32 |    1.74 M    |    0.47 M    |   27.01%   |
|  (e)  |       Same        | ResNet 110 | ResNet 56 |    1.74 M    |    0.86 M    |   49.43%   |
|  (f)  |       Same        |  WRN 40x2  | WRN 16x2  |    2.26 M    |    0.70 M    |   30.97%   |
|  (g)  |     Different     | ResNet 110 | WRN 16x1  |    1.74 M    |    0.18 M    |   10.34%   |
|  (h)  |     Different     | ResNet 110 | WRN 16x2  |    1.74 M    |    0.70 M    |   40.23%   |
|  (i)  |     Different     |  WRN 40x2  | ResNet 32 |    2.26 M    |    0.47 M    |   20.80%   |
|  (j)  |     Different     |  WRN 40x2  | ResNet 56 |    2.26 M    |    0.86 M    |   38.05%   |





### Training
Run ```CIFAR-100/SAKD_Cifar100.py``` with setting alphabet (a - f)
```
python SAKD_Cifar100.py 
--setting a 
```





### Experimental results

Performance is measured by classification accuracy in same or different net architecture (%)


| Teacher <br> Student | ResNet56 <br> ResNet20 | ResNet56 <br> ResNet32 | ResNet110 <br/> ResNet32 | ResNet110 <br/> ResNet56 | WRN40-2 <br/>WRN40-1 | WRN40-2 <br>ResNet56 | ResNet110 <br/> WRN40-1 |
| :------------------: | :--------------------: | :--------------------: | :----------------------: | :----------------------: | :------------------: | :------------------: | :---------------------: |
|       Teacher        |         73.03          |         73.03          |          74.34           |          74.34           |        75.76         |        75.76         |          74.34          |
|       Vanilla        |         69.40          |         71.14          |          71.14           |          73.03           |        71.98         |        73.03         |          66.67          |
|          KD          |         70.57          |         73.27          |          73.24           |          75.21           |        73.54         |        74.97         |          67.85          |
|        FitNet        |         70.67          |         73.31          |          73.49           |          75.34           |        72.24         |        75.06         |          67.70          |
|          AT          |         70.71          |         73.45          |          73.41           |          75.46           |        72.77         |        75.16         |          68.01          |
|         OFD          |         70.82          |         72.92          |          73.26           |          74.96           |        73.86         |        75.26         |          67.89          |
|         RKD          |         70.43          |         71.84          |          71.82           |          74.77           |        72.22         |        74.39         |          68.46          |
|         CRD          |         71.44          |         73.87          |          73.92           |          75.56           |        74.14         |        75.25         |           N/A           |
|       ReviewKD       |         71.89          |         73.34          |          73.89           |          74.83           |        75.09         |         N/A          |           N/A           |
|         SRRL         |         70.86          |          N/A           |           N/A            |           N/A            |        74.64         |         N/A          |          68.04          |
|         AFD          |         71.16          |         73.67          |          73.93           |          75.82           |         N/A          |        75.40         |           N/A           |
|        SemCKD        |          N/A           |          N/A           |           N/A            |           N/A            |        74.41         |         N/A          |           N/A           |
|        SimKD         |          N/A           |          N/A           |           N/A            |           N/A            |      **75.56**       |         N/A          |           N/A           |
|         NORM         |         71.61          |          N/A           |          73.95           |           N/A            |        75.42         |         N/A          |           N/A           |
|    **SAKD(Ours)**    |       **71.92**        |       **74.38**        |        **74.36**         |        **76.24**         |        74.97         |      **75.92**       |        **68.71**        |





### Acknowledgement

The work is based on [OFD](https://github.com/clovaai/overhaul-distillation)(ICCV 2019) / [RepDistiller](https://github.com/HobbitLong/RepDistiller)(ICLR 2020) / [AFD](https://github.com/clovaai/attention-feature-distillation)(ICCV 2021)

