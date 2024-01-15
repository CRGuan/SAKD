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

Performance is measured by classification accuracy in same net architecture (%)


| Teacher <br> Student | ResNet56 <br> ResNet20 | ResNet56 <br> ResNet32 | ResNet110 <br/> ResNet20 | ResNet110 <br/> ResNet32 | ResNet110 <br/> ResNet56 | WRN40-2 <br>WRN40-1 | WRN40-2 <br/>WRN16-2 |
| :------------------: | :--------------------: | :--------------------: | :----------------------: | :----------------------: | :----------------------: | :-----------------: | :------------------: |
|       Teacher        |         73.03          |         73.03          |          74.34           |          74.34           |          74.34           |        75.76        |        75.76         |
|       Vanilla        |         69.40          |         71.14          |          69.40           |          71.14           |          73.03           |        71.98        |        73.26         |
|          KD          |         70.57          |         73.27          |          70.81           |          73.24           |          75.21           |        73.54        |        74.92         |
|        FitNet        |         70.67          |         73.31          |          70.02           |          73.49           |          75.34           |        72.24        |        75.22         |
|          AT          |         70.71          |         73.45          |          70.81           |          73.41           |          75.46           |        72.77        |        75.20         |
|         OFD          |         70.82          |         72.92          |          70.66           |          73.26           |          74.96           |        73.86        |        75.26         |
|         RKD          |         70.43          |         71.84          |          70.76           |          71.82           |          74.77           |        72.22        |        74.59         |
|         CRD          |         71.44          |         73.87          |          70.91           |          73.92           |          75.56           |        74.14        |        75.15         |
|       ReviewKD       |         71.89          |         73.34          |           N/A            |          73.89           |          74.83           |        75.09        |        76.12         |
|         SRRL         |         70.86          |          N/A           |          70.78           |           N/A            |           N/A            |        74.64        |        75.49         |
|         AFD          |         71.16          |         73.67          |          71.38           |          73.93           |          75.82           |         N/A         |        75.47         |
|        SemCKD        |          N/A           |          N/A           |           N/A            |           N/A            |           N/A            |        74.41        |         N/A          |
|         MGD          |          N/A           |          N/A           |           N/A            |           N/A            |           N/A            |         N/A         |         N/A          |
|         NORM         |         71.61          |          N/A           |          72.00           |          73.95           |           N/A            |      **75.42**      |      **76.26**       |
|    **SAKD(Ours)**    |       **71.92**        |       **74.38**        |        **72.02**         |        **74.36**         |        **76.24**         |        75.13        |        76.02         |

Performance is measured by classification accuracy in different net architecture (%)

| Teacher <br> Student | ResNet110 <br/> WRN16-1 | ResNet110 <br/> WRN16-2 | ResNet34 <br/> WRN28-2 | ResNet50 <br/>MobileNetV2 | WRN40-2 <br/>ResNet32 | WRN40-2 <br>ResNet56 | WRN40-2 <br/>MobileNetV2 |
| :------------------: | :---------------------: | :---------------------: | :--------------------: | :-----------------------: | :-------------------: | :------------------: | :----------------------: |
|       Teacher        |          74.34          |          74.34          |         78.60          |           79.34           |         75.76         |        75.76         |          75.76           |
|       Vanilla        |          66.67          |          73.62          |         75.32          |           64.60           |         71.14         |        73.03         |          64.60           |
|          KD          |          67.78          |          75.11          |         76.48          |           67.35           |         72.91         |        74.97         |          68.70           |
|        FitNet        |          67.85          |          75.15          |         76.44          |           63.16           |         73.13         |        75.06         |          68.64           |
|          AT          |          67.70          |          74.99          |         77.20          |           58.58           |         73.22         |        75.18         |          68.79           |
|         OFD          |          68.01          |          75.22          |         76.86          |           64.75           |         73.44         |        75.26         |          68.61           |
|         RKD          |          67.89          |          75.19          |         76.32          |           64.43           |         73.51         |        74.39         |          68.71           |
|         CRD          |          68.46          |          75.66          |         76.97          |           69.11           |         73.88         |        75.25         |          70.06           |
|       ReviewKD       |           N/A           |           N/A           |          N/A           |           69.89           |          N/A          |         N/A          |           N/A            |
|         SRRL         |           N/A           |           N/A           |          N/A           |            N/A            |          N/A          |         N/A          |          69.45           |
|         AFD          |          68.04          |          75.70          |         77.47          |            N/A            |         73.94         |        75.40         |           N/A            |
|        SemCKD        |           N/A           |           N/A           |          N/A           |            N/A            |          N/A          |         N/A          |          69.61           |
|         MGD          |           N/A           |           N/A           |          N/A           |            N/A            |          N/A          |         N/A          |          68.55           |
|         NORM         |           N/A           |           N/A           |          N/A           |         **71.17**         |          N/A          |         N/A          |           N/A            |
|    **SAKD(Ours)**    |        **68.71**        |        **75.97**        |       **77.69**        |           69.92           |       **74.02**       |      **75.92**       |        **70.21**         |



### Acknowledgement

The work is based on [OFD](https://github.com/clovaai/overhaul-distillation)(ICCV 2019) / [RepDistiller](https://github.com/HobbitLong/RepDistiller)(ICLR 2020) / [AFD](https://github.com/clovaai/attention-feature-distillation)(ICCV 2021)

