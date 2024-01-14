ImageNet-1k_Resnet18_from_Resnet34_log

Teacher Net: 
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (4): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (5): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
Student Net: 
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
the number of teacher model parameters: 21797672
the number of student model parameters: 11689512
Teacher network performance
Test (on val set): [Epoch 0/100][Batch 0/196]	Time 3.144 (3.144)	Loss 0.5336 (0.5336)	Top 1-err 16.7969 (16.7969)	Top 5-err 2.3438 (2.3438)

Epoch: [0/100]	 Top 1-err 26.688  Top 5-err 8.580	 Test Loss 1.081



* Train with distillation: [Epoch 1/100][Batch 0/5005]	 Loss 14.525, Top 1-error 100.000, Top 5-error 100.000
  Train with distillation: [Epoch 1/100][Batch 500/5005]	 Loss 12.220, Top 1-error 97.575, Top 5-error 91.546
  Train with distillation: [Epoch 1/100][Batch 1000/5005]	 Loss 11.279, Top 1-error 95.090, Top 5-error 85.224
  Train with distillation: [Epoch 1/100][Batch 1500/5005]	 Loss 10.612, Top 1-error 92.620, Top 5-error 79.955
  Train with distillation: [Epoch 1/100][Batch 2000/5005]	 Loss 10.088, Top 1-error 90.309, Top 5-error 75.549
  Train with distillation: [Epoch 1/100][Batch 2500/5005]	 Loss 9.656, Top 1-error 88.204, Top 5-error 71.841
  Train with distillation: [Epoch 1/100][Batch 3000/5005]	 Loss 9.293, Top 1-error 86.323, Top 5-error 68.712
  Train with distillation: [Epoch 1/100][Batch 3500/5005]	 Loss 8.983, Top 1-error 84.623, Top 5-error 66.034
  Train with distillation: [Epoch 1/100][Batch 4000/5005]	 Loss 8.708, Top 1-error 83.063, Top 5-error 63.655
  Train with distillation: [Epoch 1/100][Batch 4500/5005]	 Loss 8.468, Top 1-error 81.647, Top 5-error 61.580
  Train with distillation: [Epoch 1/100][Batch 5000/5005]	 Loss 8.250, Top 1-error 80.360, Top 5-error 59.736
  Train 	 Time Taken: 3065.24 sec
  Test (on val set): [Epoch 1/100][Batch 0/196]	Time 1.669 (1.669)	Loss 1.9371 (1.9371)	Top 1-err 54.2969 (54.2969)	Top 5-err 21.0938 (21.0938)
* Epoch: [1/100]	 Top 1-err 68.088  Top 5-err 41.142	 Test Loss 3.153
  Current best accuracy (top-1 and 5 error): 68.088 41.142
  Train with distillation: [Epoch 2/100][Batch 0/5005]	 Loss 6.196, Top 1-error 70.312, Top 5-error 43.750
  Train with distillation: [Epoch 2/100][Batch 500/5005]	 Loss 6.070, Top 1-error 67.106, Top 5-error 41.172
  Train with distillation: [Epoch 2/100][Batch 1000/5005]	 Loss 5.991, Top 1-error 66.348, Top 5-error 40.502
  Train with distillation: [Epoch 2/100][Batch 1500/5005]	 Loss 5.920, Top 1-error 65.855, Top 5-error 40.006
  Train with distillation: [Epoch 2/100][Batch 2000/5005]	 Loss 5.854, Top 1-error 65.436, Top 5-error 39.479
  Train with distillation: [Epoch 2/100][Batch 2500/5005]	 Loss 5.794, Top 1-error 64.969, Top 5-error 39.015
  Train with distillation: [Epoch 2/100][Batch 3000/5005]	 Loss 5.737, Top 1-error 64.548, Top 5-error 38.632
  Train with distillation: [Epoch 2/100][Batch 3500/5005]	 Loss 5.683, Top 1-error 64.148, Top 5-error 38.251
  Train with distillation: [Epoch 2/100][Batch 4000/5005]	 Loss 5.633, Top 1-error 63.784, Top 5-error 37.889
  Train with distillation: [Epoch 2/100][Batch 4500/5005]	 Loss 5.585, Top 1-error 63.425, Top 5-error 37.540
  Train with distillation: [Epoch 2/100][Batch 5000/5005]	 Loss 5.540, Top 1-error 63.072, Top 5-error 37.213
  Train 	 Time Taken: 3120.86 sec
  Test (on val set): [Epoch 2/100][Batch 0/196]	Time 1.812 (1.812)	Loss 1.6982 (1.6982)	Top 1-err 42.5781 (42.5781)	Top 5-err 17.1875 (17.1875)
* Epoch: [2/100]	 Top 1-err 56.798  Top 5-err 29.308	 Test Loss 2.480
  Current best accuracy (top-1 and 5 error): 56.798 29.308
  Train with distillation: [Epoch 3/100][Batch 0/5005]	 Loss 5.558, Top 1-error 62.891, Top 5-error 36.328
  Train with distillation: [Epoch 3/100][Batch 500/5005]	 Loss 5.012, Top 1-error 58.736, Top 5-error 33.071
  Train with distillation: [Epoch 3/100][Batch 1000/5005]	 Loss 5.000, Top 1-error 58.638, Top 5-error 33.018
  Train with distillation: [Epoch 3/100][Batch 1500/5005]	 Loss 4.983, Top 1-error 58.567, Top 5-error 32.888
  Train with distillation: [Epoch 3/100][Batch 2000/5005]	 Loss 4.963, Top 1-error 58.478, Top 5-error 32.787
  Train with distillation: [Epoch 3/100][Batch 2500/5005]	 Loss 4.937, Top 1-error 58.284, Top 5-error 32.587
  Train with distillation: [Epoch 3/100][Batch 3000/5005]	 Loss 4.916, Top 1-error 58.133, Top 5-error 32.416
  Train with distillation: [Epoch 3/100][Batch 3500/5005]	 Loss 4.896, Top 1-error 57.999, Top 5-error 32.293
  Train with distillation: [Epoch 3/100][Batch 4000/5005]	 Loss 4.876, Top 1-error 57.829, Top 5-error 32.151
  Train with distillation: [Epoch 3/100][Batch 4500/5005]	 Loss 4.857, Top 1-error 57.708, Top 5-error 32.025
  Train with distillation: [Epoch 3/100][Batch 5000/5005]	 Loss 4.837, Top 1-error 57.564, Top 5-error 31.884
  Train 	 Time Taken: 3116.14 sec
  Test (on val set): [Epoch 3/100][Batch 0/196]	Time 1.838 (1.838)	Loss 1.3692 (1.3692)	Top 1-err 36.3281 (36.3281)	Top 5-err 10.9375 (10.9375)
* Epoch: [3/100]	 Top 1-err 53.184  Top 5-err 26.394	 Test Loss 2.322
  Current best accuracy (top-1 and 5 error): 53.184 26.394
  Train with distillation: [Epoch 4/100][Batch 0/5005]	 Loss 4.546, Top 1-error 59.375, Top 5-error 30.469
  Train with distillation: [Epoch 4/100][Batch 500/5005]	 Loss 4.566, Top 1-error 55.166, Top 5-error 29.712
  Train with distillation: [Epoch 4/100][Batch 1000/5005]	 Loss 4.572, Top 1-error 55.288, Top 5-error 29.791
  Train with distillation: [Epoch 4/100][Batch 1500/5005]	 Loss 4.566, Top 1-error 55.178, Top 5-error 29.761
  Train with distillation: [Epoch 4/100][Batch 2000/5005]	 Loss 4.560, Top 1-error 55.233, Top 5-error 29.752
  Train with distillation: [Epoch 4/100][Batch 2500/5005]	 Loss 4.551, Top 1-error 55.188, Top 5-error 29.737
  Train with distillation: [Epoch 4/100][Batch 3000/5005]	 Loss 4.540, Top 1-error 55.140, Top 5-error 29.670
  Train with distillation: [Epoch 4/100][Batch 3500/5005]	 Loss 4.533, Top 1-error 55.080, Top 5-error 29.652
  Train with distillation: [Epoch 4/100][Batch 4000/5005]	 Loss 4.523, Top 1-error 55.005, Top 5-error 29.578
  Train with distillation: [Epoch 4/100][Batch 4500/5005]	 Loss 4.514, Top 1-error 54.940, Top 5-error 29.520
  Train with distillation: [Epoch 4/100][Batch 5000/5005]	 Loss 4.503, Top 1-error 54.844, Top 5-error 29.433
  Train 	 Time Taken: 3112.20 sec
  Test (on val set): [Epoch 4/100][Batch 0/196]	Time 2.076 (2.076)	Loss 1.3786 (1.3786)	Top 1-err 37.8906 (37.8906)	Top 5-err 10.1562 (10.1562)
* Epoch: [4/100]	 Top 1-err 52.034  Top 5-err 25.284	 Test Loss 2.246
  Current best accuracy (top-1 and 5 error): 52.034 25.284
  Train with distillation: [Epoch 5/100][Batch 0/5005]	 Loss 4.438, Top 1-error 57.031, Top 5-error 28.125
  Train with distillation: [Epoch 5/100][Batch 500/5005]	 Loss 4.336, Top 1-error 53.214, Top 5-error 28.224
  Train with distillation: [Epoch 5/100][Batch 1000/5005]	 Loss 4.344, Top 1-error 53.338, Top 5-error 28.271
  Train with distillation: [Epoch 5/100][Batch 1500/5005]	 Loss 4.345, Top 1-error 53.468, Top 5-error 28.348
  Train with distillation: [Epoch 5/100][Batch 2000/5005]	 Loss 4.342, Top 1-error 53.442, Top 5-error 28.326
  Train with distillation: [Epoch 5/100][Batch 2500/5005]	 Loss 4.337, Top 1-error 53.397, Top 5-error 28.262
  Train with distillation: [Epoch 5/100][Batch 3000/5005]	 Loss 4.332, Top 1-error 53.359, Top 5-error 28.216
  Train with distillation: [Epoch 5/100][Batch 3500/5005]	 Loss 4.327, Top 1-error 53.348, Top 5-error 28.195
  Train with distillation: [Epoch 5/100][Batch 4000/5005]	 Loss 4.323, Top 1-error 53.344, Top 5-error 28.181
  Train with distillation: [Epoch 5/100][Batch 4500/5005]	 Loss 4.319, Top 1-error 53.300, Top 5-error 28.126
  Train with distillation: [Epoch 5/100][Batch 5000/5005]	 Loss 4.313, Top 1-error 53.285, Top 5-error 28.107
  Train 	 Time Taken: 3108.74 sec
  Test (on val set): [Epoch 5/100][Batch 0/196]	Time 1.837 (1.837)	Loss 1.4042 (1.4042)	Top 1-err 38.2812 (38.2812)	Top 5-err 9.3750 (9.3750)
* Epoch: [5/100]	 Top 1-err 50.196  Top 5-err 24.042	 Test Loss 2.163
  Current best accuracy (top-1 and 5 error): 50.196 24.042
  Train with distillation: [Epoch 6/100][Batch 0/5005]	 Loss 3.996, Top 1-error 48.828, Top 5-error 23.047
  Train with distillation: [Epoch 6/100][Batch 500/5005]	 Loss 4.206, Top 1-error 52.325, Top 5-error 27.167
  Train with distillation: [Epoch 6/100][Batch 1000/5005]	 Loss 4.206, Top 1-error 52.340, Top 5-error 27.224
  Train with distillation: [Epoch 6/100][Batch 1500/5005]	 Loss 4.204, Top 1-error 52.307, Top 5-error 27.224
  Train with distillation: [Epoch 6/100][Batch 2000/5005]	 Loss 4.200, Top 1-error 52.298, Top 5-error 27.243
  Train with distillation: [Epoch 6/100][Batch 2500/5005]	 Loss 4.199, Top 1-error 52.334, Top 5-error 27.257
  Train with distillation: [Epoch 6/100][Batch 3000/5005]	 Loss 4.196, Top 1-error 52.316, Top 5-error 27.233
  Train with distillation: [Epoch 6/100][Batch 3500/5005]	 Loss 4.194, Top 1-error 52.313, Top 5-error 27.229
  Train with distillation: [Epoch 6/100][Batch 4000/5005]	 Loss 4.191, Top 1-error 52.302, Top 5-error 27.214
  Train with distillation: [Epoch 6/100][Batch 4500/5005]	 Loss 4.187, Top 1-error 52.300, Top 5-error 27.202
  Train with distillation: [Epoch 6/100][Batch 5000/5005]	 Loss 4.184, Top 1-error 52.274, Top 5-error 27.196
  Train 	 Time Taken: 3107.20 sec
  Test (on val set): [Epoch 6/100][Batch 0/196]	Time 2.059 (2.059)	Loss 1.4422 (1.4422)	Top 1-err 41.4062 (41.4062)	Top 5-err 14.8438 (14.8438)
* Epoch: [6/100]	 Top 1-err 50.486  Top 5-err 23.780	 Test Loss 2.183
  Current best accuracy (top-1 and 5 error): 50.196 24.042
  Train with distillation: [Epoch 7/100][Batch 0/5005]	 Loss 4.065, Top 1-error 49.219, Top 5-error 25.781
  Train with distillation: [Epoch 7/100][Batch 500/5005]	 Loss 4.092, Top 1-error 51.332, Top 5-error 26.461
  Train with distillation: [Epoch 7/100][Batch 1000/5005]	 Loss 4.094, Top 1-error 51.369, Top 5-error 26.447
  Train with distillation: [Epoch 7/100][Batch 1500/5005]	 Loss 4.098, Top 1-error 51.445, Top 5-error 26.513
  Train with distillation: [Epoch 7/100][Batch 2000/5005]	 Loss 4.099, Top 1-error 51.476, Top 5-error 26.531
  Train with distillation: [Epoch 7/100][Batch 2500/5005]	 Loss 4.098, Top 1-error 51.496, Top 5-error 26.560
  Train with distillation: [Epoch 7/100][Batch 3000/5005]	 Loss 4.097, Top 1-error 51.508, Top 5-error 26.588
  Train with distillation: [Epoch 7/100][Batch 3500/5005]	 Loss 4.098, Top 1-error 51.528, Top 5-error 26.608
  Train with distillation: [Epoch 7/100][Batch 4000/5005]	 Loss 4.096, Top 1-error 51.500, Top 5-error 26.601
  Train with distillation: [Epoch 7/100][Batch 4500/5005]	 Loss 4.095, Top 1-error 51.505, Top 5-error 26.586
  Train with distillation: [Epoch 7/100][Batch 5000/5005]	 Loss 4.093, Top 1-error 51.515, Top 5-error 26.590
  Train 	 Time Taken: 3098.71 sec
  Test (on val set): [Epoch 7/100][Batch 0/196]	Time 1.842 (1.842)	Loss 1.4320 (1.4320)	Top 1-err 35.1562 (35.1562)	Top 5-err 13.2812 (13.2812)
* Epoch: [7/100]	 Top 1-err 53.796  Top 5-err 27.322	 Test Loss 2.400
  Current best accuracy (top-1 and 5 error): 50.196 24.042
  Train with distillation: [Epoch 8/100][Batch 0/5005]	 Loss 4.038, Top 1-error 49.609, Top 5-error 25.000
  Train with distillation: [Epoch 8/100][Batch 500/5005]	 Loss 4.023, Top 1-error 50.635, Top 5-error 25.922
  Train with distillation: [Epoch 8/100][Batch 1000/5005]	 Loss 4.028, Top 1-error 50.765, Top 5-error 26.071
  Train with distillation: [Epoch 8/100][Batch 1500/5005]	 Loss 4.035, Top 1-error 50.913, Top 5-error 26.125
  Train with distillation: [Epoch 8/100][Batch 2000/5005]	 Loss 4.038, Top 1-error 50.974, Top 5-error 26.152
  Train with distillation: [Epoch 8/100][Batch 2500/5005]	 Loss 4.039, Top 1-error 50.998, Top 5-error 26.154
  Train with distillation: [Epoch 8/100][Batch 3000/5005]	 Loss 4.037, Top 1-error 50.990, Top 5-error 26.144
  Train with distillation: [Epoch 8/100][Batch 3500/5005]	 Loss 4.034, Top 1-error 50.990, Top 5-error 26.120
  Train with distillation: [Epoch 8/100][Batch 4000/5005]	 Loss 4.033, Top 1-error 51.013, Top 5-error 26.130
  Train with distillation: [Epoch 8/100][Batch 4500/5005]	 Loss 4.033, Top 1-error 51.031, Top 5-error 26.128
  Train with distillation: [Epoch 8/100][Batch 5000/5005]	 Loss 4.031, Top 1-error 51.017, Top 5-error 26.114
  Train 	 Time Taken: 3086.74 sec
  Test (on val set): [Epoch 8/100][Batch 0/196]	Time 2.059 (2.059)	Loss 1.4021 (1.4021)	Top 1-err 36.7188 (36.7188)	Top 5-err 11.3281 (11.3281)
* Epoch: [8/100]	 Top 1-err 48.264  Top 5-err 22.280	 Test Loss 2.066
  Current best accuracy (top-1 and 5 error): 48.264 22.28
  Train with distillation: [Epoch 9/100][Batch 0/5005]	 Loss 3.916, Top 1-error 48.828, Top 5-error 25.391
  Train with distillation: [Epoch 9/100][Batch 500/5005]	 Loss 3.960, Top 1-error 50.148, Top 5-error 25.491
  Train with distillation: [Epoch 9/100][Batch 1000/5005]	 Loss 3.975, Top 1-error 50.430, Top 5-error 25.647
  Train with distillation: [Epoch 9/100][Batch 1500/5005]	 Loss 3.974, Top 1-error 50.461, Top 5-error 25.641
  Train with distillation: [Epoch 9/100][Batch 2000/5005]	 Loss 3.974, Top 1-error 50.457, Top 5-error 25.628
  Train with distillation: [Epoch 9/100][Batch 2500/5005]	 Loss 3.977, Top 1-error 50.528, Top 5-error 25.686
  Train with distillation: [Epoch 9/100][Batch 3000/5005]	 Loss 3.977, Top 1-error 50.542, Top 5-error 25.700
  Train with distillation: [Epoch 9/100][Batch 3500/5005]	 Loss 3.978, Top 1-error 50.551, Top 5-error 25.719
  Train with distillation: [Epoch 9/100][Batch 4000/5005]	 Loss 3.978, Top 1-error 50.542, Top 5-error 25.734
  Train with distillation: [Epoch 9/100][Batch 4500/5005]	 Loss 3.978, Top 1-error 50.557, Top 5-error 25.753
  Train with distillation: [Epoch 9/100][Batch 5000/5005]	 Loss 3.977, Top 1-error 50.555, Top 5-error 25.752
  Train 	 Time Taken: 3090.49 sec
  Test (on val set): [Epoch 9/100][Batch 0/196]	Time 1.866 (1.866)	Loss 1.2432 (1.2432)	Top 1-err 33.5938 (33.5938)	Top 5-err 10.9375 (10.9375)
* Epoch: [9/100]	 Top 1-err 48.890  Top 5-err 22.620	 Test Loss 2.086
  Current best accuracy (top-1 and 5 error): 48.264 22.28
  Train with distillation: [Epoch 10/100][Batch 0/5005]	 Loss 4.012, Top 1-error 51.953, Top 5-error 29.297
  Train with distillation: [Epoch 10/100][Batch 500/5005]	 Loss 3.896, Top 1-error 49.580, Top 5-error 25.068
  Train with distillation: [Epoch 10/100][Batch 1000/5005]	 Loss 3.914, Top 1-error 49.951, Top 5-error 25.224
  Train with distillation: [Epoch 10/100][Batch 1500/5005]	 Loss 3.923, Top 1-error 50.014, Top 5-error 25.310
  Train with distillation: [Epoch 10/100][Batch 2000/5005]	 Loss 3.928, Top 1-error 50.104, Top 5-error 25.346
  Train with distillation: [Epoch 10/100][Batch 2500/5005]	 Loss 3.931, Top 1-error 50.153, Top 5-error 25.406
  Train with distillation: [Epoch 10/100][Batch 3000/5005]	 Loss 3.932, Top 1-error 50.171, Top 5-error 25.426
  Train with distillation: [Epoch 10/100][Batch 3500/5005]	 Loss 3.933, Top 1-error 50.194, Top 5-error 25.458
  Train with distillation: [Epoch 10/100][Batch 4000/5005]	 Loss 3.933, Top 1-error 50.181, Top 5-error 25.462
  Train with distillation: [Epoch 10/100][Batch 4500/5005]	 Loss 3.934, Top 1-error 50.213, Top 5-error 25.487
  Train with distillation: [Epoch 10/100][Batch 5000/5005]	 Loss 3.934, Top 1-error 50.216, Top 5-error 25.491
  Train 	 Time Taken: 3082.06 sec
  Test (on val set): [Epoch 10/100][Batch 0/196]	Time 2.076 (2.076)	Loss 1.1518 (1.1518)	Top 1-err 35.5469 (35.5469)	Top 5-err 7.8125 (7.8125)
* Epoch: [10/100]	 Top 1-err 50.818  Top 5-err 24.406	 Test Loss 2.212
  Current best accuracy (top-1 and 5 error): 48.264 22.28
  Train with distillation: [Epoch 11/100][Batch 0/5005]	 Loss 3.854, Top 1-error 51.953, Top 5-error 25.000
  Train with distillation: [Epoch 11/100][Batch 500/5005]	 Loss 3.881, Top 1-error 49.541, Top 5-error 24.905
  Train with distillation: [Epoch 11/100][Batch 1000/5005]	 Loss 3.887, Top 1-error 49.637, Top 5-error 24.953
  Train with distillation: [Epoch 11/100][Batch 1500/5005]	 Loss 3.894, Top 1-error 49.841, Top 5-error 25.066
  Train with distillation: [Epoch 11/100][Batch 2000/5005]	 Loss 3.898, Top 1-error 49.872, Top 5-error 25.151
  Train with distillation: [Epoch 11/100][Batch 2500/5005]	 Loss 3.897, Top 1-error 49.852, Top 5-error 25.160
  Train with distillation: [Epoch 11/100][Batch 3000/5005]	 Loss 3.900, Top 1-error 49.890, Top 5-error 25.209
  Train with distillation: [Epoch 11/100][Batch 3500/5005]	 Loss 3.902, Top 1-error 49.927, Top 5-error 25.239
  Train with distillation: [Epoch 11/100][Batch 4000/5005]	 Loss 3.902, Top 1-error 49.942, Top 5-error 25.243
  Train with distillation: [Epoch 11/100][Batch 4500/5005]	 Loss 3.902, Top 1-error 49.941, Top 5-error 25.254
  Train with distillation: [Epoch 11/100][Batch 5000/5005]	 Loss 3.902, Top 1-error 49.945, Top 5-error 25.257
  Train 	 Time Taken: 3081.83 sec
  Test (on val set): [Epoch 11/100][Batch 0/196]	Time 1.854 (1.854)	Loss 1.3192 (1.3192)	Top 1-err 34.7656 (34.7656)	Top 5-err 9.7656 (9.7656)
* Epoch: [11/100]	 Top 1-err 47.690  Top 5-err 21.680	 Test Loss 2.021
  Current best accuracy (top-1 and 5 error): 47.69 21.68
  Train with distillation: [Epoch 12/100][Batch 0/5005]	 Loss 4.035, Top 1-error 50.781, Top 5-error 26.172
  Train with distillation: [Epoch 12/100][Batch 500/5005]	 Loss 3.823, Top 1-error 49.162, Top 5-error 24.482
  Train with distillation: [Epoch 12/100][Batch 1000/5005]	 Loss 3.847, Top 1-error 49.366, Top 5-error 24.761
  Train with distillation: [Epoch 12/100][Batch 1500/5005]	 Loss 3.858, Top 1-error 49.488, Top 5-error 24.848
  Train with distillation: [Epoch 12/100][Batch 2000/5005]	 Loss 3.859, Top 1-error 49.540, Top 5-error 24.850
  Train with distillation: [Epoch 12/100][Batch 2500/5005]	 Loss 3.864, Top 1-error 49.618, Top 5-error 24.907
  Train with distillation: [Epoch 12/100][Batch 3000/5005]	 Loss 3.864, Top 1-error 49.643, Top 5-error 24.909
  Train with distillation: [Epoch 12/100][Batch 3500/5005]	 Loss 3.869, Top 1-error 49.685, Top 5-error 24.964
  Train with distillation: [Epoch 12/100][Batch 4000/5005]	 Loss 3.869, Top 1-error 49.655, Top 5-error 24.970
  Train with distillation: [Epoch 12/100][Batch 4500/5005]	 Loss 3.869, Top 1-error 49.677, Top 5-error 24.985
  Train with distillation: [Epoch 12/100][Batch 5000/5005]	 Loss 3.869, Top 1-error 49.696, Top 5-error 24.995
  Train 	 Time Taken: 3079.41 sec
  Test (on val set): [Epoch 12/100][Batch 0/196]	Time 2.058 (2.058)	Loss 1.2928 (1.2928)	Top 1-err 38.2812 (38.2812)	Top 5-err 8.9844 (8.9844)
* Epoch: [12/100]	 Top 1-err 46.634  Top 5-err 20.812	 Test Loss 1.972
  Current best accuracy (top-1 and 5 error): 46.634 20.812
  Train with distillation: [Epoch 13/100][Batch 0/5005]	 Loss 3.555, Top 1-error 41.797, Top 5-error 18.359
  Train with distillation: [Epoch 13/100][Batch 500/5005]	 Loss 3.809, Top 1-error 48.728, Top 5-error 24.414
  Train with distillation: [Epoch 13/100][Batch 1000/5005]	 Loss 3.818, Top 1-error 48.886, Top 5-error 24.509
  Train with distillation: [Epoch 13/100][Batch 1500/5005]	 Loss 3.824, Top 1-error 49.065, Top 5-error 24.639
  Train with distillation: [Epoch 13/100][Batch 2000/5005]	 Loss 3.829, Top 1-error 49.152, Top 5-error 24.683
  Train with distillation: [Epoch 13/100][Batch 2500/5005]	 Loss 3.831, Top 1-error 49.181, Top 5-error 24.703
  Train with distillation: [Epoch 13/100][Batch 3000/5005]	 Loss 3.837, Top 1-error 49.306, Top 5-error 24.793
  Train with distillation: [Epoch 13/100][Batch 3500/5005]	 Loss 3.841, Top 1-error 49.385, Top 5-error 24.824
  Train with distillation: [Epoch 13/100][Batch 4000/5005]	 Loss 3.841, Top 1-error 49.387, Top 5-error 24.838
  Train with distillation: [Epoch 13/100][Batch 4500/5005]	 Loss 3.843, Top 1-error 49.428, Top 5-error 24.860
  Train with distillation: [Epoch 13/100][Batch 5000/5005]	 Loss 3.844, Top 1-error 49.430, Top 5-error 24.856
  Train 	 Time Taken: 3087.89 sec
  Test (on val set): [Epoch 13/100][Batch 0/196]	Time 1.867 (1.867)	Loss 1.3112 (1.3112)	Top 1-err 35.1562 (35.1562)	Top 5-err 12.5000 (12.5000)
* Epoch: [13/100]	 Top 1-err 47.036  Top 5-err 21.182	 Test Loss 1.984
  Current best accuracy (top-1 and 5 error): 46.634 20.812
  Train with distillation: [Epoch 14/100][Batch 0/5005]	 Loss 4.062, Top 1-error 51.562, Top 5-error 25.000
  Train with distillation: [Epoch 14/100][Batch 500/5005]	 Loss 3.794, Top 1-error 48.780, Top 5-error 24.462
  Train with distillation: [Epoch 14/100][Batch 1000/5005]	 Loss 3.807, Top 1-error 49.005, Top 5-error 24.560
  Train with distillation: [Epoch 14/100][Batch 1500/5005]	 Loss 3.809, Top 1-error 49.091, Top 5-error 24.557
  Train with distillation: [Epoch 14/100][Batch 2000/5005]	 Loss 3.816, Top 1-error 49.167, Top 5-error 24.608
  Train with distillation: [Epoch 14/100][Batch 2500/5005]	 Loss 3.819, Top 1-error 49.197, Top 5-error 24.664
  Train with distillation: [Epoch 14/100][Batch 3000/5005]	 Loss 3.824, Top 1-error 49.223, Top 5-error 24.730
  Train with distillation: [Epoch 14/100][Batch 3500/5005]	 Loss 3.826, Top 1-error 49.275, Top 5-error 24.771
  Train with distillation: [Epoch 14/100][Batch 4000/5005]	 Loss 3.828, Top 1-error 49.292, Top 5-error 24.771
  Train with distillation: [Epoch 14/100][Batch 4500/5005]	 Loss 3.828, Top 1-error 49.305, Top 5-error 24.776
  Train with distillation: [Epoch 14/100][Batch 5000/5005]	 Loss 3.828, Top 1-error 49.325, Top 5-error 24.795
  Train 	 Time Taken: 3124.42 sec
  Test (on val set): [Epoch 14/100][Batch 0/196]	Time 2.066 (2.066)	Loss 1.2260 (1.2260)	Top 1-err 32.4219 (32.4219)	Top 5-err 9.3750 (9.3750)
* Epoch: [14/100]	 Top 1-err 47.736  Top 5-err 21.900	 Test Loss 2.040
  Current best accuracy (top-1 and 5 error): 46.634 20.812
  Train with distillation: [Epoch 15/100][Batch 0/5005]	 Loss 3.762, Top 1-error 46.875, Top 5-error 26.172
  Train with distillation: [Epoch 15/100][Batch 500/5005]	 Loss 3.765, Top 1-error 48.705, Top 5-error 24.283
  Train with distillation: [Epoch 15/100][Batch 1000/5005]	 Loss 3.782, Top 1-error 48.910, Top 5-error 24.394
  Train with distillation: [Epoch 15/100][Batch 1500/5005]	 Loss 3.787, Top 1-error 48.940, Top 5-error 24.422
  Train with distillation: [Epoch 15/100][Batch 2000/5005]	 Loss 3.791, Top 1-error 48.949, Top 5-error 24.463
  Train with distillation: [Epoch 15/100][Batch 2500/5005]	 Loss 3.796, Top 1-error 49.013, Top 5-error 24.546
  Train with distillation: [Epoch 15/100][Batch 3000/5005]	 Loss 3.799, Top 1-error 49.059, Top 5-error 24.581
  Train with distillation: [Epoch 15/100][Batch 3500/5005]	 Loss 3.802, Top 1-error 49.100, Top 5-error 24.611
  Train with distillation: [Epoch 15/100][Batch 4000/5005]	 Loss 3.803, Top 1-error 49.085, Top 5-error 24.634
  Train with distillation: [Epoch 15/100][Batch 4500/5005]	 Loss 3.805, Top 1-error 49.128, Top 5-error 24.677
  Train with distillation: [Epoch 15/100][Batch 5000/5005]	 Loss 3.805, Top 1-error 49.129, Top 5-error 24.668
  Train 	 Time Taken: 3131.45 sec
  Test (on val set): [Epoch 15/100][Batch 0/196]	Time 1.828 (1.828)	Loss 1.2244 (1.2244)	Top 1-err 32.8125 (32.8125)	Top 5-err 10.9375 (10.9375)
* Epoch: [15/100]	 Top 1-err 46.602  Top 5-err 21.130	 Test Loss 1.960
  Current best accuracy (top-1 and 5 error): 46.602 21.13
  Train with distillation: [Epoch 16/100][Batch 0/5005]	 Loss 3.741, Top 1-error 46.484, Top 5-error 25.000
  Train with distillation: [Epoch 16/100][Batch 500/5005]	 Loss 3.774, Top 1-error 48.579, Top 5-error 24.213
  Train with distillation: [Epoch 16/100][Batch 1000/5005]	 Loss 3.771, Top 1-error 48.637, Top 5-error 24.240
  Train with distillation: [Epoch 16/100][Batch 1500/5005]	 Loss 3.777, Top 1-error 48.725, Top 5-error 24.327
  Train with distillation: [Epoch 16/100][Batch 2000/5005]	 Loss 3.778, Top 1-error 48.778, Top 5-error 24.317
  Train with distillation: [Epoch 16/100][Batch 2500/5005]	 Loss 3.779, Top 1-error 48.812, Top 5-error 24.346
  Train with distillation: [Epoch 16/100][Batch 3000/5005]	 Loss 3.781, Top 1-error 48.851, Top 5-error 24.357
  Train with distillation: [Epoch 16/100][Batch 3500/5005]	 Loss 3.784, Top 1-error 48.914, Top 5-error 24.414
  Train with distillation: [Epoch 16/100][Batch 4000/5005]	 Loss 3.787, Top 1-error 48.927, Top 5-error 24.442
  Train with distillation: [Epoch 16/100][Batch 4500/5005]	 Loss 3.788, Top 1-error 48.952, Top 5-error 24.463
  Train with distillation: [Epoch 16/100][Batch 5000/5005]	 Loss 3.790, Top 1-error 48.985, Top 5-error 24.495
  Train 	 Time Taken: 3152.14 sec
  Test (on val set): [Epoch 16/100][Batch 0/196]	Time 2.078 (2.078)	Loss 1.1732 (1.1732)	Top 1-err 32.8125 (32.8125)	Top 5-err 10.9375 (10.9375)
* Epoch: [16/100]	 Top 1-err 47.146  Top 5-err 21.516	 Test Loss 1.996
  Current best accuracy (top-1 and 5 error): 46.602 21.13
  Train with distillation: [Epoch 17/100][Batch 0/5005]	 Loss 3.882, Top 1-error 49.609, Top 5-error 25.781
  Train with distillation: [Epoch 17/100][Batch 500/5005]	 Loss 3.736, Top 1-error 48.255, Top 5-error 23.961
  Train with distillation: [Epoch 17/100][Batch 1000/5005]	 Loss 3.748, Top 1-error 48.461, Top 5-error 24.118
  Train with distillation: [Epoch 17/100][Batch 1500/5005]	 Loss 3.754, Top 1-error 48.528, Top 5-error 24.178
  Train with distillation: [Epoch 17/100][Batch 2000/5005]	 Loss 3.758, Top 1-error 48.603, Top 5-error 24.211
  Train with distillation: [Epoch 17/100][Batch 2500/5005]	 Loss 3.765, Top 1-error 48.694, Top 5-error 24.295
  Train with distillation: [Epoch 17/100][Batch 3000/5005]	 Loss 3.767, Top 1-error 48.748, Top 5-error 24.319
  Train with distillation: [Epoch 17/100][Batch 3500/5005]	 Loss 3.770, Top 1-error 48.775, Top 5-error 24.351
  Train with distillation: [Epoch 17/100][Batch 4000/5005]	 Loss 3.772, Top 1-error 48.821, Top 5-error 24.378
  Train with distillation: [Epoch 17/100][Batch 4500/5005]	 Loss 3.773, Top 1-error 48.812, Top 5-error 24.380
  Train with distillation: [Epoch 17/100][Batch 5000/5005]	 Loss 3.775, Top 1-error 48.842, Top 5-error 24.415
  Train 	 Time Taken: 3164.92 sec
  Test (on val set): [Epoch 17/100][Batch 0/196]	Time 1.851 (1.851)	Loss 1.1845 (1.1845)	Top 1-err 33.9844 (33.9844)	Top 5-err 8.5938 (8.5938)
* Epoch: [17/100]	 Top 1-err 46.150  Top 5-err 20.678	 Test Loss 1.955
  Current best accuracy (top-1 and 5 error): 46.15 20.678
  Train with distillation: [Epoch 18/100][Batch 0/5005]	 Loss 3.931, Top 1-error 51.172, Top 5-error 24.609
  Train with distillation: [Epoch 18/100][Batch 500/5005]	 Loss 3.731, Top 1-error 48.530, Top 5-error 23.905
  Train with distillation: [Epoch 18/100][Batch 1000/5005]	 Loss 3.736, Top 1-error 48.493, Top 5-error 23.947
  Train with distillation: [Epoch 18/100][Batch 1500/5005]	 Loss 3.738, Top 1-error 48.525, Top 5-error 24.018
  Train with distillation: [Epoch 18/100][Batch 2000/5005]	 Loss 3.744, Top 1-error 48.613, Top 5-error 24.098
  Train with distillation: [Epoch 18/100][Batch 2500/5005]	 Loss 3.752, Top 1-error 48.681, Top 5-error 24.174
  Train with distillation: [Epoch 18/100][Batch 3000/5005]	 Loss 3.753, Top 1-error 48.668, Top 5-error 24.199
  Train with distillation: [Epoch 18/100][Batch 3500/5005]	 Loss 3.756, Top 1-error 48.689, Top 5-error 24.243
  Train with distillation: [Epoch 18/100][Batch 4000/5005]	 Loss 3.757, Top 1-error 48.714, Top 5-error 24.267
  Train with distillation: [Epoch 18/100][Batch 4500/5005]	 Loss 3.758, Top 1-error 48.718, Top 5-error 24.276
  Train with distillation: [Epoch 18/100][Batch 5000/5005]	 Loss 3.759, Top 1-error 48.719, Top 5-error 24.291
  Train 	 Time Taken: 3167.33 sec
  Test (on val set): [Epoch 18/100][Batch 0/196]	Time 2.026 (2.026)	Loss 1.2743 (1.2743)	Top 1-err 32.4219 (32.4219)	Top 5-err 11.3281 (11.3281)
* Epoch: [18/100]	 Top 1-err 47.298  Top 5-err 21.670	 Test Loss 2.020
  Current best accuracy (top-1 and 5 error): 46.15 20.678
  Train with distillation: [Epoch 19/100][Batch 0/5005]	 Loss 3.804, Top 1-error 50.000, Top 5-error 24.219
  Train with distillation: [Epoch 19/100][Batch 500/5005]	 Loss 3.722, Top 1-error 48.395, Top 5-error 23.989
  Train with distillation: [Epoch 19/100][Batch 1000/5005]	 Loss 3.727, Top 1-error 48.395, Top 5-error 23.963
  Train with distillation: [Epoch 19/100][Batch 1500/5005]	 Loss 3.736, Top 1-error 48.531, Top 5-error 24.090
  Train with distillation: [Epoch 19/100][Batch 2000/5005]	 Loss 3.739, Top 1-error 48.560, Top 5-error 24.115
  Train with distillation: [Epoch 19/100][Batch 2500/5005]	 Loss 3.743, Top 1-error 48.585, Top 5-error 24.166
  Train with distillation: [Epoch 19/100][Batch 3000/5005]	 Loss 3.745, Top 1-error 48.605, Top 5-error 24.189
  Train with distillation: [Epoch 19/100][Batch 3500/5005]	 Loss 3.745, Top 1-error 48.618, Top 5-error 24.174
  Train with distillation: [Epoch 19/100][Batch 4000/5005]	 Loss 3.747, Top 1-error 48.650, Top 5-error 24.205
  Train with distillation: [Epoch 19/100][Batch 4500/5005]	 Loss 3.750, Top 1-error 48.693, Top 5-error 24.225
  Train with distillation: [Epoch 19/100][Batch 5000/5005]	 Loss 3.751, Top 1-error 48.717, Top 5-error 24.246
  Train 	 Time Taken: 3179.81 sec
  Test (on val set): [Epoch 19/100][Batch 0/196]	Time 1.906 (1.906)	Loss 1.1937 (1.1937)	Top 1-err 33.2031 (33.2031)	Top 5-err 9.7656 (9.7656)
* Epoch: [19/100]	 Top 1-err 46.080  Top 5-err 20.700	 Test Loss 1.947
  Current best accuracy (top-1 and 5 error): 46.08 20.7
  Train with distillation: [Epoch 20/100][Batch 0/5005]	 Loss 3.643, Top 1-error 49.219, Top 5-error 22.656
  Train with distillation: [Epoch 20/100][Batch 500/5005]	 Loss 3.716, Top 1-error 48.293, Top 5-error 23.982
  Train with distillation: [Epoch 20/100][Batch 1000/5005]	 Loss 3.722, Top 1-error 48.283, Top 5-error 23.995
  Train with distillation: [Epoch 20/100][Batch 1500/5005]	 Loss 3.726, Top 1-error 48.325, Top 5-error 24.055
  Train with distillation: [Epoch 20/100][Batch 2000/5005]	 Loss 3.729, Top 1-error 48.340, Top 5-error 24.075
  Train with distillation: [Epoch 20/100][Batch 2500/5005]	 Loss 3.730, Top 1-error 48.373, Top 5-error 24.059
  Train with distillation: [Epoch 20/100][Batch 3000/5005]	 Loss 3.731, Top 1-error 48.413, Top 5-error 24.055
  Train with distillation: [Epoch 20/100][Batch 3500/5005]	 Loss 3.732, Top 1-error 48.433, Top 5-error 24.067
  Train with distillation: [Epoch 20/100][Batch 4000/5005]	 Loss 3.735, Top 1-error 48.478, Top 5-error 24.116
  Train with distillation: [Epoch 20/100][Batch 4500/5005]	 Loss 3.737, Top 1-error 48.506, Top 5-error 24.146
  Train with distillation: [Epoch 20/100][Batch 5000/5005]	 Loss 3.738, Top 1-error 48.519, Top 5-error 24.173
  Train 	 Time Taken: 3190.58 sec
  Test (on val set): [Epoch 20/100][Batch 0/196]	Time 2.052 (2.052)	Loss 1.4383 (1.4383)	Top 1-err 41.0156 (41.0156)	Top 5-err 11.7188 (11.7188)
* Epoch: [20/100]	 Top 1-err 45.748  Top 5-err 20.594	 Test Loss 1.939
  Current best accuracy (top-1 and 5 error): 45.748 20.594
  Train with distillation: [Epoch 21/100][Batch 0/5005]	 Loss 3.572, Top 1-error 44.922, Top 5-error 22.656
  Train with distillation: [Epoch 21/100][Batch 500/5005]	 Loss 3.719, Top 1-error 48.220, Top 5-error 23.972
  Train with distillation: [Epoch 21/100][Batch 1000/5005]	 Loss 3.711, Top 1-error 48.282, Top 5-error 23.925
  Train with distillation: [Epoch 21/100][Batch 1500/5005]	 Loss 3.709, Top 1-error 48.253, Top 5-error 23.902
  Train with distillation: [Epoch 21/100][Batch 2000/5005]	 Loss 3.716, Top 1-error 48.296, Top 5-error 23.976
  Train with distillation: [Epoch 21/100][Batch 2500/5005]	 Loss 3.720, Top 1-error 48.298, Top 5-error 23.981
  Train with distillation: [Epoch 21/100][Batch 3000/5005]	 Loss 3.722, Top 1-error 48.343, Top 5-error 24.011
  Train with distillation: [Epoch 21/100][Batch 3500/5005]	 Loss 3.723, Top 1-error 48.378, Top 5-error 24.036
  Train with distillation: [Epoch 21/100][Batch 4000/5005]	 Loss 3.725, Top 1-error 48.372, Top 5-error 24.029
  Train with distillation: [Epoch 21/100][Batch 4500/5005]	 Loss 3.727, Top 1-error 48.413, Top 5-error 24.059
  Train with distillation: [Epoch 21/100][Batch 5000/5005]	 Loss 3.729, Top 1-error 48.449, Top 5-error 24.095
  Train 	 Time Taken: 3189.08 sec
  Test (on val set): [Epoch 21/100][Batch 0/196]	Time 1.968 (1.968)	Loss 1.2714 (1.2714)	Top 1-err 32.8125 (32.8125)	Top 5-err 10.9375 (10.9375)
* Epoch: [21/100]	 Top 1-err 46.318  Top 5-err 20.478	 Test Loss 1.947
  Current best accuracy (top-1 and 5 error): 45.748 20.594
  Train with distillation: [Epoch 22/100][Batch 0/5005]	 Loss 3.716, Top 1-error 48.438, Top 5-error 22.266
  Train with distillation: [Epoch 22/100][Batch 500/5005]	 Loss 3.686, Top 1-error 47.971, Top 5-error 23.728
  Train with distillation: [Epoch 22/100][Batch 1000/5005]	 Loss 3.684, Top 1-error 47.960, Top 5-error 23.703
  Train with distillation: [Epoch 22/100][Batch 1500/5005]	 Loss 3.692, Top 1-error 48.152, Top 5-error 23.794
  Train with distillation: [Epoch 22/100][Batch 2000/5005]	 Loss 3.701, Top 1-error 48.214, Top 5-error 23.872
  Train with distillation: [Epoch 22/100][Batch 2500/5005]	 Loss 3.704, Top 1-error 48.216, Top 5-error 23.900
  Train with distillation: [Epoch 22/100][Batch 3000/5005]	 Loss 3.710, Top 1-error 48.272, Top 5-error 23.938
  Train with distillation: [Epoch 22/100][Batch 3500/5005]	 Loss 3.712, Top 1-error 48.303, Top 5-error 23.957
  Train with distillation: [Epoch 22/100][Batch 4000/5005]	 Loss 3.714, Top 1-error 48.344, Top 5-error 23.986
  Train with distillation: [Epoch 22/100][Batch 4500/5005]	 Loss 3.715, Top 1-error 48.358, Top 5-error 23.976
  Train with distillation: [Epoch 22/100][Batch 5000/5005]	 Loss 3.716, Top 1-error 48.384, Top 5-error 23.993
  Train 	 Time Taken: 3187.91 sec
  Test (on val set): [Epoch 22/100][Batch 0/196]	Time 2.026 (2.026)	Loss 1.0801 (1.0801)	Top 1-err 29.2969 (29.2969)	Top 5-err 7.8125 (7.8125)
* Epoch: [22/100]	 Top 1-err 45.192  Top 5-err 19.744	 Test Loss 1.893
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 23/100][Batch 0/5005]	 Loss 3.807, Top 1-error 49.219, Top 5-error 27.734
  Train with distillation: [Epoch 23/100][Batch 500/5005]	 Loss 3.672, Top 1-error 47.855, Top 5-error 23.430
  Train with distillation: [Epoch 23/100][Batch 1000/5005]	 Loss 3.682, Top 1-error 48.027, Top 5-error 23.635
  Train with distillation: [Epoch 23/100][Batch 1500/5005]	 Loss 3.691, Top 1-error 48.093, Top 5-error 23.757
  Train with distillation: [Epoch 23/100][Batch 2000/5005]	 Loss 3.694, Top 1-error 48.150, Top 5-error 23.818
  Train with distillation: [Epoch 23/100][Batch 2500/5005]	 Loss 3.700, Top 1-error 48.226, Top 5-error 23.896
  Train with distillation: [Epoch 23/100][Batch 3000/5005]	 Loss 3.703, Top 1-error 48.251, Top 5-error 23.895
  Train with distillation: [Epoch 23/100][Batch 3500/5005]	 Loss 3.705, Top 1-error 48.263, Top 5-error 23.911
  Train with distillation: [Epoch 23/100][Batch 4000/5005]	 Loss 3.708, Top 1-error 48.267, Top 5-error 23.944
  Train with distillation: [Epoch 23/100][Batch 4500/5005]	 Loss 3.710, Top 1-error 48.296, Top 5-error 23.960
  Train with distillation: [Epoch 23/100][Batch 5000/5005]	 Loss 3.710, Top 1-error 48.301, Top 5-error 23.962
  Train 	 Time Taken: 3172.37 sec
  Test (on val set): [Epoch 23/100][Batch 0/196]	Time 1.881 (1.881)	Loss 1.2945 (1.2945)	Top 1-err 40.2344 (40.2344)	Top 5-err 8.5938 (8.5938)
* Epoch: [23/100]	 Top 1-err 45.702  Top 5-err 20.078	 Test Loss 1.919
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 24/100][Batch 0/5005]	 Loss 3.745, Top 1-error 45.703, Top 5-error 22.266
  Train with distillation: [Epoch 24/100][Batch 500/5005]	 Loss 3.678, Top 1-error 47.817, Top 5-error 23.677
  Train with distillation: [Epoch 24/100][Batch 1000/5005]	 Loss 3.686, Top 1-error 47.960, Top 5-error 23.796
  Train with distillation: [Epoch 24/100][Batch 1500/5005]	 Loss 3.689, Top 1-error 48.015, Top 5-error 23.811
  Train with distillation: [Epoch 24/100][Batch 2000/5005]	 Loss 3.691, Top 1-error 48.067, Top 5-error 23.853
  Train with distillation: [Epoch 24/100][Batch 2500/5005]	 Loss 3.695, Top 1-error 48.131, Top 5-error 23.885
  Train with distillation: [Epoch 24/100][Batch 3000/5005]	 Loss 3.699, Top 1-error 48.158, Top 5-error 23.900
  Train with distillation: [Epoch 24/100][Batch 3500/5005]	 Loss 3.701, Top 1-error 48.190, Top 5-error 23.928
  Train with distillation: [Epoch 24/100][Batch 4000/5005]	 Loss 3.702, Top 1-error 48.208, Top 5-error 23.956
  Train with distillation: [Epoch 24/100][Batch 4500/5005]	 Loss 3.703, Top 1-error 48.217, Top 5-error 23.963
  Train with distillation: [Epoch 24/100][Batch 5000/5005]	 Loss 3.705, Top 1-error 48.265, Top 5-error 23.988
  Train 	 Time Taken: 3163.06 sec
  Test (on val set): [Epoch 24/100][Batch 0/196]	Time 1.959 (1.959)	Loss 1.0589 (1.0589)	Top 1-err 32.0312 (32.0312)	Top 5-err 7.8125 (7.8125)
* Epoch: [24/100]	 Top 1-err 45.508  Top 5-err 20.060	 Test Loss 1.909
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 25/100][Batch 0/5005]	 Loss 3.667, Top 1-error 49.219, Top 5-error 25.391
  Train with distillation: [Epoch 25/100][Batch 500/5005]	 Loss 3.677, Top 1-error 47.924, Top 5-error 23.736
  Train with distillation: [Epoch 25/100][Batch 1000/5005]	 Loss 3.678, Top 1-error 47.930, Top 5-error 23.676
  Train with distillation: [Epoch 25/100][Batch 1500/5005]	 Loss 3.682, Top 1-error 47.938, Top 5-error 23.718
  Train with distillation: [Epoch 25/100][Batch 2000/5005]	 Loss 3.680, Top 1-error 47.910, Top 5-error 23.704
  Train with distillation: [Epoch 25/100][Batch 2500/5005]	 Loss 3.686, Top 1-error 47.985, Top 5-error 23.780
  Train with distillation: [Epoch 25/100][Batch 3000/5005]	 Loss 3.688, Top 1-error 48.022, Top 5-error 23.813
  Train with distillation: [Epoch 25/100][Batch 3500/5005]	 Loss 3.692, Top 1-error 48.075, Top 5-error 23.857
  Train with distillation: [Epoch 25/100][Batch 4000/5005]	 Loss 3.695, Top 1-error 48.131, Top 5-error 23.875
  Train with distillation: [Epoch 25/100][Batch 4500/5005]	 Loss 3.697, Top 1-error 48.155, Top 5-error 23.881
  Train with distillation: [Epoch 25/100][Batch 5000/5005]	 Loss 3.699, Top 1-error 48.187, Top 5-error 23.906
  Train 	 Time Taken: 3157.76 sec
  Test (on val set): [Epoch 25/100][Batch 0/196]	Time 1.898 (1.898)	Loss 1.1863 (1.1863)	Top 1-err 34.7656 (34.7656)	Top 5-err 7.0312 (7.0312)
* Epoch: [25/100]	 Top 1-err 46.464  Top 5-err 20.934	 Test Loss 1.976
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 26/100][Batch 0/5005]	 Loss 3.666, Top 1-error 49.609, Top 5-error 24.609
  Train with distillation: [Epoch 26/100][Batch 500/5005]	 Loss 3.669, Top 1-error 47.817, Top 5-error 23.576
  Train with distillation: [Epoch 26/100][Batch 1000/5005]	 Loss 3.673, Top 1-error 47.800, Top 5-error 23.603
  Train with distillation: [Epoch 26/100][Batch 1500/5005]	 Loss 3.679, Top 1-error 47.892, Top 5-error 23.681
  Train with distillation: [Epoch 26/100][Batch 2000/5005]	 Loss 3.681, Top 1-error 47.940, Top 5-error 23.687
  Train with distillation: [Epoch 26/100][Batch 2500/5005]	 Loss 3.683, Top 1-error 47.976, Top 5-error 23.715
  Train with distillation: [Epoch 26/100][Batch 3000/5005]	 Loss 3.686, Top 1-error 48.027, Top 5-error 23.743
  Train with distillation: [Epoch 26/100][Batch 3500/5005]	 Loss 3.687, Top 1-error 48.075, Top 5-error 23.753
  Train with distillation: [Epoch 26/100][Batch 4000/5005]	 Loss 3.688, Top 1-error 48.099, Top 5-error 23.775
  Train with distillation: [Epoch 26/100][Batch 4500/5005]	 Loss 3.689, Top 1-error 48.083, Top 5-error 23.795
  Train with distillation: [Epoch 26/100][Batch 5000/5005]	 Loss 3.691, Top 1-error 48.106, Top 5-error 23.831
  Train 	 Time Taken: 3143.57 sec
  Test (on val set): [Epoch 26/100][Batch 0/196]	Time 1.982 (1.982)	Loss 1.4036 (1.4036)	Top 1-err 35.1562 (35.1562)	Top 5-err 10.9375 (10.9375)
* Epoch: [26/100]	 Top 1-err 45.710  Top 5-err 20.458	 Test Loss 1.933
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 27/100][Batch 0/5005]	 Loss 3.491, Top 1-error 46.484, Top 5-error 17.969
  Train with distillation: [Epoch 27/100][Batch 500/5005]	 Loss 3.645, Top 1-error 47.707, Top 5-error 23.359
  Train with distillation: [Epoch 27/100][Batch 1000/5005]	 Loss 3.659, Top 1-error 47.761, Top 5-error 23.540
  Train with distillation: [Epoch 27/100][Batch 1500/5005]	 Loss 3.665, Top 1-error 47.894, Top 5-error 23.590
  Train with distillation: [Epoch 27/100][Batch 2000/5005]	 Loss 3.671, Top 1-error 47.947, Top 5-error 23.653
  Train with distillation: [Epoch 27/100][Batch 2500/5005]	 Loss 3.674, Top 1-error 48.021, Top 5-error 23.727
  Train with distillation: [Epoch 27/100][Batch 3000/5005]	 Loss 3.677, Top 1-error 48.079, Top 5-error 23.758
  Train with distillation: [Epoch 27/100][Batch 3500/5005]	 Loss 3.678, Top 1-error 48.069, Top 5-error 23.764
  Train with distillation: [Epoch 27/100][Batch 4000/5005]	 Loss 3.678, Top 1-error 48.068, Top 5-error 23.770
  Train with distillation: [Epoch 27/100][Batch 4500/5005]	 Loss 3.681, Top 1-error 48.095, Top 5-error 23.798
  Train with distillation: [Epoch 27/100][Batch 5000/5005]	 Loss 3.683, Top 1-error 48.132, Top 5-error 23.816
  Train 	 Time Taken: 3127.06 sec
  Test (on val set): [Epoch 27/100][Batch 0/196]	Time 1.913 (1.913)	Loss 0.9729 (0.9729)	Top 1-err 28.1250 (28.1250)	Top 5-err 7.8125 (7.8125)
* Epoch: [27/100]	 Top 1-err 46.230  Top 5-err 20.454	 Test Loss 1.951
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 28/100][Batch 0/5005]	 Loss 3.459, Top 1-error 43.359, Top 5-error 20.312
  Train with distillation: [Epoch 28/100][Batch 500/5005]	 Loss 3.651, Top 1-error 47.744, Top 5-error 23.597
  Train with distillation: [Epoch 28/100][Batch 1000/5005]	 Loss 3.663, Top 1-error 47.874, Top 5-error 23.709
  Train with distillation: [Epoch 28/100][Batch 1500/5005]	 Loss 3.671, Top 1-error 47.988, Top 5-error 23.754
  Train with distillation: [Epoch 28/100][Batch 2000/5005]	 Loss 3.674, Top 1-error 48.025, Top 5-error 23.748
  Train with distillation: [Epoch 28/100][Batch 2500/5005]	 Loss 3.672, Top 1-error 47.984, Top 5-error 23.745
  Train with distillation: [Epoch 28/100][Batch 3000/5005]	 Loss 3.674, Top 1-error 48.014, Top 5-error 23.771
  Train with distillation: [Epoch 28/100][Batch 3500/5005]	 Loss 3.675, Top 1-error 48.038, Top 5-error 23.753
  Train with distillation: [Epoch 28/100][Batch 4000/5005]	 Loss 3.677, Top 1-error 48.056, Top 5-error 23.770
  Train with distillation: [Epoch 28/100][Batch 4500/5005]	 Loss 3.679, Top 1-error 48.101, Top 5-error 23.793
  Train with distillation: [Epoch 28/100][Batch 5000/5005]	 Loss 3.680, Top 1-error 48.118, Top 5-error 23.796
  Train 	 Time Taken: 3115.33 sec
  Test (on val set): [Epoch 28/100][Batch 0/196]	Time 2.002 (2.002)	Loss 1.2945 (1.2945)	Top 1-err 30.4688 (30.4688)	Top 5-err 12.1094 (12.1094)
* Epoch: [28/100]	 Top 1-err 46.026  Top 5-err 20.642	 Test Loss 1.952
  Current best accuracy (top-1 and 5 error): 45.192 19.744
  Train with distillation: [Epoch 29/100][Batch 0/5005]	 Loss 3.678, Top 1-error 46.875, Top 5-error 23.438
  Train with distillation: [Epoch 29/100][Batch 500/5005]	 Loss 3.630, Top 1-error 47.538, Top 5-error 23.335
  Train with distillation: [Epoch 29/100][Batch 1000/5005]	 Loss 3.641, Top 1-error 47.609, Top 5-error 23.388
  Train with distillation: [Epoch 29/100][Batch 1500/5005]	 Loss 3.646, Top 1-error 47.656, Top 5-error 23.421
  Train with distillation: [Epoch 29/100][Batch 2000/5005]	 Loss 3.652, Top 1-error 47.743, Top 5-error 23.506
  Train with distillation: [Epoch 29/100][Batch 2500/5005]	 Loss 3.656, Top 1-error 47.754, Top 5-error 23.537
  Train with distillation: [Epoch 29/100][Batch 3000/5005]	 Loss 3.662, Top 1-error 47.828, Top 5-error 23.581
  Train with distillation: [Epoch 29/100][Batch 3500/5005]	 Loss 3.665, Top 1-error 47.869, Top 5-error 23.633
  Train with distillation: [Epoch 29/100][Batch 4000/5005]	 Loss 3.667, Top 1-error 47.915, Top 5-error 23.671
  Train with distillation: [Epoch 29/100][Batch 4500/5005]	 Loss 3.669, Top 1-error 47.933, Top 5-error 23.692
  Train with distillation: [Epoch 29/100][Batch 5000/5005]	 Loss 3.671, Top 1-error 47.950, Top 5-error 23.707
  Train 	 Time Taken: 3107.50 sec
  Test (on val set): [Epoch 29/100][Batch 0/196]	Time 1.925 (1.925)	Loss 1.5860 (1.5860)	Top 1-err 41.7969 (41.7969)	Top 5-err 12.5000 (12.5000)
* Epoch: [29/100]	 Top 1-err 44.634  Top 5-err 19.388	 Test Loss 1.873
  Current best accuracy (top-1 and 5 error): 44.634 19.388
  Train with distillation: [Epoch 30/100][Batch 0/5005]	 Loss 3.635, Top 1-error 45.703, Top 5-error 21.484
  Train with distillation: [Epoch 30/100][Batch 500/5005]	 Loss 3.096, Top 1-error 42.482, Top 5-error 19.888
  Train with distillation: [Epoch 30/100][Batch 1000/5005]	 Loss 3.011, Top 1-error 41.684, Top 5-error 19.288
  Train with distillation: [Epoch 30/100][Batch 1500/5005]	 Loss 2.960, Top 1-error 41.190, Top 5-error 18.962
  Train with distillation: [Epoch 30/100][Batch 2000/5005]	 Loss 2.927, Top 1-error 40.848, Top 5-error 18.775
  Train with distillation: [Epoch 30/100][Batch 2500/5005]	 Loss 2.900, Top 1-error 40.630, Top 5-error 18.624
  Train with distillation: [Epoch 30/100][Batch 3000/5005]	 Loss 2.876, Top 1-error 40.427, Top 5-error 18.466
  Train with distillation: [Epoch 30/100][Batch 3500/5005]	 Loss 2.857, Top 1-error 40.244, Top 5-error 18.326
  Train with distillation: [Epoch 30/100][Batch 4000/5005]	 Loss 2.841, Top 1-error 40.099, Top 5-error 18.239
  Train with distillation: [Epoch 30/100][Batch 4500/5005]	 Loss 2.826, Top 1-error 39.957, Top 5-error 18.148
  Train with distillation: [Epoch 30/100][Batch 5000/5005]	 Loss 2.813, Top 1-error 39.848, Top 5-error 18.080
  Train 	 Time Taken: 3095.74 sec
  Test (on val set): [Epoch 30/100][Batch 0/196]	Time 2.026 (2.026)	Loss 0.7930 (0.7930)	Top 1-err 23.0469 (23.0469)	Top 5-err 4.2969 (4.2969)
* Epoch: [30/100]	 Top 1-err 34.466  Top 5-err 12.792	 Test Loss 1.391
  Current best accuracy (top-1 and 5 error): 34.466 12.792
  Train with distillation: [Epoch 31/100][Batch 0/5005]	 Loss 2.618, Top 1-error 37.109, Top 5-error 14.453
  Train with distillation: [Epoch 31/100][Batch 500/5005]	 Loss 2.656, Top 1-error 38.202, Top 5-error 16.803
  Train with distillation: [Epoch 31/100][Batch 1000/5005]	 Loss 2.654, Top 1-error 38.083, Top 5-error 16.875
  Train with distillation: [Epoch 31/100][Batch 1500/5005]	 Loss 2.655, Top 1-error 38.137, Top 5-error 16.904
  Train with distillation: [Epoch 31/100][Batch 2000/5005]	 Loss 2.650, Top 1-error 38.117, Top 5-error 16.881
  Train with distillation: [Epoch 31/100][Batch 2500/5005]	 Loss 2.648, Top 1-error 38.113, Top 5-error 16.855
  Train with distillation: [Epoch 31/100][Batch 3000/5005]	 Loss 2.644, Top 1-error 38.066, Top 5-error 16.814
  Train with distillation: [Epoch 31/100][Batch 3500/5005]	 Loss 2.641, Top 1-error 38.037, Top 5-error 16.810
  Train with distillation: [Epoch 31/100][Batch 4000/5005]	 Loss 2.637, Top 1-error 37.995, Top 5-error 16.778
  Train with distillation: [Epoch 31/100][Batch 4500/5005]	 Loss 2.635, Top 1-error 37.950, Top 5-error 16.755
  Train with distillation: [Epoch 31/100][Batch 5000/5005]	 Loss 2.632, Top 1-error 37.957, Top 5-error 16.752
  Train 	 Time Taken: 3089.88 sec
  Test (on val set): [Epoch 31/100][Batch 0/196]	Time 1.907 (1.907)	Loss 0.7421 (0.7421)	Top 1-err 20.3125 (20.3125)	Top 5-err 5.0781 (5.0781)
* Epoch: [31/100]	 Top 1-err 33.444  Top 5-err 12.316	 Test Loss 1.351
  Current best accuracy (top-1 and 5 error): 33.444 12.316
  Train with distillation: [Epoch 32/100][Batch 0/5005]	 Loss 2.692, Top 1-error 42.969, Top 5-error 19.141
  Train with distillation: [Epoch 32/100][Batch 500/5005]	 Loss 2.573, Top 1-error 37.174, Top 5-error 16.263
  Train with distillation: [Epoch 32/100][Batch 1000/5005]	 Loss 2.577, Top 1-error 37.324, Top 5-error 16.332
  Train with distillation: [Epoch 32/100][Batch 1500/5005]	 Loss 2.578, Top 1-error 37.355, Top 5-error 16.356
  Train with distillation: [Epoch 32/100][Batch 2000/5005]	 Loss 2.575, Top 1-error 37.295, Top 5-error 16.332
  Train with distillation: [Epoch 32/100][Batch 2500/5005]	 Loss 2.573, Top 1-error 37.280, Top 5-error 16.324
  Train with distillation: [Epoch 32/100][Batch 3000/5005]	 Loss 2.572, Top 1-error 37.273, Top 5-error 16.313
  Train with distillation: [Epoch 32/100][Batch 3500/5005]	 Loss 2.572, Top 1-error 37.299, Top 5-error 16.319
  Train with distillation: [Epoch 32/100][Batch 4000/5005]	 Loss 2.572, Top 1-error 37.316, Top 5-error 16.320
  Train with distillation: [Epoch 32/100][Batch 4500/5005]	 Loss 2.571, Top 1-error 37.322, Top 5-error 16.310
  Train with distillation: [Epoch 32/100][Batch 5000/5005]	 Loss 2.569, Top 1-error 37.310, Top 5-error 16.302
  Train 	 Time Taken: 3081.86 sec
  Test (on val set): [Epoch 32/100][Batch 0/196]	Time 2.043 (2.043)	Loss 0.7302 (0.7302)	Top 1-err 21.0938 (21.0938)	Top 5-err 3.9062 (3.9062)
* Epoch: [32/100]	 Top 1-err 33.596  Top 5-err 12.212	 Test Loss 1.345
  Current best accuracy (top-1 and 5 error): 33.444 12.316
  Train with distillation: [Epoch 33/100][Batch 0/5005]	 Loss 2.473, Top 1-error 33.203, Top 5-error 12.891
  Train with distillation: [Epoch 33/100][Batch 500/5005]	 Loss 2.526, Top 1-error 36.751, Top 5-error 15.919
  Train with distillation: [Epoch 33/100][Batch 1000/5005]	 Loss 2.534, Top 1-error 36.873, Top 5-error 16.022
  Train with distillation: [Epoch 33/100][Batch 1500/5005]	 Loss 2.535, Top 1-error 36.887, Top 5-error 16.021
  Train with distillation: [Epoch 33/100][Batch 2000/5005]	 Loss 2.532, Top 1-error 36.854, Top 5-error 16.001
  Train with distillation: [Epoch 33/100][Batch 2500/5005]	 Loss 2.532, Top 1-error 36.879, Top 5-error 16.006
  Train with distillation: [Epoch 33/100][Batch 3000/5005]	 Loss 2.530, Top 1-error 36.839, Top 5-error 15.980
  Train with distillation: [Epoch 33/100][Batch 3500/5005]	 Loss 2.529, Top 1-error 36.835, Top 5-error 15.977
  Train with distillation: [Epoch 33/100][Batch 4000/5005]	 Loss 2.528, Top 1-error 36.828, Top 5-error 15.971
  Train with distillation: [Epoch 33/100][Batch 4500/5005]	 Loss 2.529, Top 1-error 36.838, Top 5-error 15.981
  Train with distillation: [Epoch 33/100][Batch 5000/5005]	 Loss 2.527, Top 1-error 36.809, Top 5-error 15.976
  Train 	 Time Taken: 3078.06 sec
  Test (on val set): [Epoch 33/100][Batch 0/196]	Time 1.939 (1.939)	Loss 0.7257 (0.7257)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.9062 (3.9062)
* Epoch: [33/100]	 Top 1-err 33.042  Top 5-err 11.714	 Test Loss 1.318
  Current best accuracy (top-1 and 5 error): 33.042 11.714
  Train with distillation: [Epoch 34/100][Batch 0/5005]	 Loss 2.502, Top 1-error 41.016, Top 5-error 15.625
  Train with distillation: [Epoch 34/100][Batch 500/5005]	 Loss 2.496, Top 1-error 36.424, Top 5-error 15.609
  Train with distillation: [Epoch 34/100][Batch 1000/5005]	 Loss 2.496, Top 1-error 36.428, Top 5-error 15.688
  Train with distillation: [Epoch 34/100][Batch 1500/5005]	 Loss 2.495, Top 1-error 36.385, Top 5-error 15.691
  Train with distillation: [Epoch 34/100][Batch 2000/5005]	 Loss 2.499, Top 1-error 36.442, Top 5-error 15.735
  Train with distillation: [Epoch 34/100][Batch 2500/5005]	 Loss 2.501, Top 1-error 36.466, Top 5-error 15.757
  Train with distillation: [Epoch 34/100][Batch 3000/5005]	 Loss 2.501, Top 1-error 36.463, Top 5-error 15.778
  Train with distillation: [Epoch 34/100][Batch 3500/5005]	 Loss 2.502, Top 1-error 36.479, Top 5-error 15.786
  Train with distillation: [Epoch 34/100][Batch 4000/5005]	 Loss 2.503, Top 1-error 36.499, Top 5-error 15.783
  Train with distillation: [Epoch 34/100][Batch 4500/5005]	 Loss 2.503, Top 1-error 36.503, Top 5-error 15.785
  Train with distillation: [Epoch 34/100][Batch 5000/5005]	 Loss 2.504, Top 1-error 36.526, Top 5-error 15.817
  Train 	 Time Taken: 3090.67 sec
  Test (on val set): [Epoch 34/100][Batch 0/196]	Time 2.026 (2.026)	Loss 0.7350 (0.7350)	Top 1-err 20.3125 (20.3125)	Top 5-err 3.9062 (3.9062)
* Epoch: [34/100]	 Top 1-err 32.628  Top 5-err 11.696	 Test Loss 1.312
  Current best accuracy (top-1 and 5 error): 32.628 11.696
  Train with distillation: [Epoch 35/100][Batch 0/5005]	 Loss 2.492, Top 1-error 33.984, Top 5-error 15.234
  Train with distillation: [Epoch 35/100][Batch 500/5005]	 Loss 2.472, Top 1-error 36.012, Top 5-error 15.547
  Train with distillation: [Epoch 35/100][Batch 1000/5005]	 Loss 2.482, Top 1-error 36.132, Top 5-error 15.673
  Train with distillation: [Epoch 35/100][Batch 1500/5005]	 Loss 2.482, Top 1-error 36.171, Top 5-error 15.629
  Train with distillation: [Epoch 35/100][Batch 2000/5005]	 Loss 2.481, Top 1-error 36.213, Top 5-error 15.591
  Train with distillation: [Epoch 35/100][Batch 2500/5005]	 Loss 2.481, Top 1-error 36.234, Top 5-error 15.602
  Train with distillation: [Epoch 35/100][Batch 3000/5005]	 Loss 2.481, Top 1-error 36.244, Top 5-error 15.623
  Train with distillation: [Epoch 35/100][Batch 3500/5005]	 Loss 2.482, Top 1-error 36.269, Top 5-error 15.618
  Train with distillation: [Epoch 35/100][Batch 4000/5005]	 Loss 2.484, Top 1-error 36.305, Top 5-error 15.642
  Train with distillation: [Epoch 35/100][Batch 4500/5005]	 Loss 2.485, Top 1-error 36.322, Top 5-error 15.649
  Train with distillation: [Epoch 35/100][Batch 5000/5005]	 Loss 2.486, Top 1-error 36.356, Top 5-error 15.658
  Train 	 Time Taken: 3089.08 sec
  Test (on val set): [Epoch 35/100][Batch 0/196]	Time 1.926 (1.926)	Loss 0.7314 (0.7314)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.9062 (3.9062)
* Epoch: [35/100]	 Top 1-err 32.794  Top 5-err 11.682	 Test Loss 1.315
  Current best accuracy (top-1 and 5 error): 32.628 11.696
  Train with distillation: [Epoch 36/100][Batch 0/5005]	 Loss 2.359, Top 1-error 33.984, Top 5-error 14.062
  Train with distillation: [Epoch 36/100][Batch 500/5005]	 Loss 2.467, Top 1-error 35.860, Top 5-error 15.382
  Train with distillation: [Epoch 36/100][Batch 1000/5005]	 Loss 2.471, Top 1-error 35.971, Top 5-error 15.458
  Train with distillation: [Epoch 36/100][Batch 1500/5005]	 Loss 2.471, Top 1-error 36.048, Top 5-error 15.467
  Train with distillation: [Epoch 36/100][Batch 2000/5005]	 Loss 2.471, Top 1-error 36.049, Top 5-error 15.480
  Train with distillation: [Epoch 36/100][Batch 2500/5005]	 Loss 2.474, Top 1-error 36.093, Top 5-error 15.538
  Train with distillation: [Epoch 36/100][Batch 3000/5005]	 Loss 2.473, Top 1-error 36.067, Top 5-error 15.518
  Train with distillation: [Epoch 36/100][Batch 3500/5005]	 Loss 2.474, Top 1-error 36.081, Top 5-error 15.541
  Train with distillation: [Epoch 36/100][Batch 4000/5005]	 Loss 2.476, Top 1-error 36.122, Top 5-error 15.573
  Train with distillation: [Epoch 36/100][Batch 4500/5005]	 Loss 2.476, Top 1-error 36.128, Top 5-error 15.572
  Train with distillation: [Epoch 36/100][Batch 5000/5005]	 Loss 2.477, Top 1-error 36.154, Top 5-error 15.587
  Train 	 Time Taken: 3083.11 sec
  Test (on val set): [Epoch 36/100][Batch 0/196]	Time 2.032 (2.032)	Loss 0.7988 (0.7988)	Top 1-err 22.2656 (22.2656)	Top 5-err 4.2969 (4.2969)
* Epoch: [36/100]	 Top 1-err 32.428  Top 5-err 11.576	 Test Loss 1.308
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 37/100][Batch 0/5005]	 Loss 2.736, Top 1-error 40.625, Top 5-error 18.359
  Train with distillation: [Epoch 37/100][Batch 500/5005]	 Loss 2.464, Top 1-error 35.850, Top 5-error 15.404
  Train with distillation: [Epoch 37/100][Batch 1000/5005]	 Loss 2.461, Top 1-error 35.785, Top 5-error 15.327
  Train with distillation: [Epoch 37/100][Batch 1500/5005]	 Loss 2.461, Top 1-error 35.827, Top 5-error 15.380
  Train with distillation: [Epoch 37/100][Batch 2000/5005]	 Loss 2.464, Top 1-error 35.875, Top 5-error 15.413
  Train with distillation: [Epoch 37/100][Batch 2500/5005]	 Loss 2.464, Top 1-error 35.914, Top 5-error 15.413
  Train with distillation: [Epoch 37/100][Batch 3000/5005]	 Loss 2.466, Top 1-error 35.939, Top 5-error 15.430
  Train with distillation: [Epoch 37/100][Batch 3500/5005]	 Loss 2.467, Top 1-error 35.944, Top 5-error 15.419
  Train with distillation: [Epoch 37/100][Batch 4000/5005]	 Loss 2.469, Top 1-error 35.981, Top 5-error 15.439
  Train with distillation: [Epoch 37/100][Batch 4500/5005]	 Loss 2.470, Top 1-error 36.012, Top 5-error 15.454
  Train with distillation: [Epoch 37/100][Batch 5000/5005]	 Loss 2.471, Top 1-error 36.038, Top 5-error 15.475
  Train 	 Time Taken: 3072.45 sec
  Test (on val set): [Epoch 37/100][Batch 0/196]	Time 1.943 (1.943)	Loss 0.7252 (0.7252)	Top 1-err 18.3594 (18.3594)	Top 5-err 5.0781 (5.0781)
* Epoch: [37/100]	 Top 1-err 32.804  Top 5-err 11.734	 Test Loss 1.312
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 38/100][Batch 0/5005]	 Loss 2.457, Top 1-error 35.547, Top 5-error 15.625
  Train with distillation: [Epoch 38/100][Batch 500/5005]	 Loss 2.451, Top 1-error 35.648, Top 5-error 15.333
  Train with distillation: [Epoch 38/100][Batch 1000/5005]	 Loss 2.451, Top 1-error 35.650, Top 5-error 15.264
  Train with distillation: [Epoch 38/100][Batch 1500/5005]	 Loss 2.454, Top 1-error 35.680, Top 5-error 15.309
  Train with distillation: [Epoch 38/100][Batch 2000/5005]	 Loss 2.460, Top 1-error 35.776, Top 5-error 15.346
  Train with distillation: [Epoch 38/100][Batch 2500/5005]	 Loss 2.462, Top 1-error 35.823, Top 5-error 15.361
  Train with distillation: [Epoch 38/100][Batch 3000/5005]	 Loss 2.464, Top 1-error 35.880, Top 5-error 15.392
  Train with distillation: [Epoch 38/100][Batch 3500/5005]	 Loss 2.464, Top 1-error 35.869, Top 5-error 15.378
  Train with distillation: [Epoch 38/100][Batch 4000/5005]	 Loss 2.465, Top 1-error 35.911, Top 5-error 15.395
  Train with distillation: [Epoch 38/100][Batch 4500/5005]	 Loss 2.466, Top 1-error 35.905, Top 5-error 15.396
  Train with distillation: [Epoch 38/100][Batch 5000/5005]	 Loss 2.468, Top 1-error 35.941, Top 5-error 15.430
  Train 	 Time Taken: 3065.18 sec
  Test (on val set): [Epoch 38/100][Batch 0/196]	Time 2.024 (2.024)	Loss 0.7279 (0.7279)	Top 1-err 21.0938 (21.0938)	Top 5-err 4.2969 (4.2969)
* Epoch: [38/100]	 Top 1-err 32.788  Top 5-err 11.648	 Test Loss 1.312
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 39/100][Batch 0/5005]	 Loss 2.484, Top 1-error 36.328, Top 5-error 16.406
  Train with distillation: [Epoch 39/100][Batch 500/5005]	 Loss 2.449, Top 1-error 35.623, Top 5-error 15.022
  Train with distillation: [Epoch 39/100][Batch 1000/5005]	 Loss 2.450, Top 1-error 35.605, Top 5-error 15.143
  Train with distillation: [Epoch 39/100][Batch 1500/5005]	 Loss 2.453, Top 1-error 35.672, Top 5-error 15.154
  Train with distillation: [Epoch 39/100][Batch 2000/5005]	 Loss 2.456, Top 1-error 35.709, Top 5-error 15.187
  Train with distillation: [Epoch 39/100][Batch 2500/5005]	 Loss 2.458, Top 1-error 35.728, Top 5-error 15.220
  Train with distillation: [Epoch 39/100][Batch 3000/5005]	 Loss 2.461, Top 1-error 35.790, Top 5-error 15.276
  Train with distillation: [Epoch 39/100][Batch 3500/5005]	 Loss 2.463, Top 1-error 35.790, Top 5-error 15.297
  Train with distillation: [Epoch 39/100][Batch 4000/5005]	 Loss 2.465, Top 1-error 35.808, Top 5-error 15.312
  Train with distillation: [Epoch 39/100][Batch 4500/5005]	 Loss 2.466, Top 1-error 35.844, Top 5-error 15.332
  Train with distillation: [Epoch 39/100][Batch 5000/5005]	 Loss 2.468, Top 1-error 35.871, Top 5-error 15.350
  Train 	 Time Taken: 3052.38 sec
  Test (on val set): [Epoch 39/100][Batch 0/196]	Time 1.904 (1.904)	Loss 0.7589 (0.7589)	Top 1-err 20.7031 (20.7031)	Top 5-err 4.2969 (4.2969)
* Epoch: [39/100]	 Top 1-err 32.522  Top 5-err 11.800	 Test Loss 1.306
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 40/100][Batch 0/5005]	 Loss 2.625, Top 1-error 40.234, Top 5-error 17.578
  Train with distillation: [Epoch 40/100][Batch 500/5005]	 Loss 2.449, Top 1-error 35.685, Top 5-error 15.206
  Train with distillation: [Epoch 40/100][Batch 1000/5005]	 Loss 2.449, Top 1-error 35.666, Top 5-error 15.187
  Train with distillation: [Epoch 40/100][Batch 1500/5005]	 Loss 2.449, Top 1-error 35.640, Top 5-error 15.166
  Train with distillation: [Epoch 40/100][Batch 2000/5005]	 Loss 2.456, Top 1-error 35.742, Top 5-error 15.253
  Train with distillation: [Epoch 40/100][Batch 2500/5005]	 Loss 2.461, Top 1-error 35.819, Top 5-error 15.308
  Train with distillation: [Epoch 40/100][Batch 3000/5005]	 Loss 2.464, Top 1-error 35.863, Top 5-error 15.329
  Train with distillation: [Epoch 40/100][Batch 3500/5005]	 Loss 2.466, Top 1-error 35.890, Top 5-error 15.365
  Train with distillation: [Epoch 40/100][Batch 4000/5005]	 Loss 2.469, Top 1-error 35.933, Top 5-error 15.401
  Train with distillation: [Epoch 40/100][Batch 4500/5005]	 Loss 2.470, Top 1-error 35.955, Top 5-error 15.397
  Train with distillation: [Epoch 40/100][Batch 5000/5005]	 Loss 2.471, Top 1-error 35.960, Top 5-error 15.406
  Train 	 Time Taken: 3052.79 sec
  Test (on val set): [Epoch 40/100][Batch 0/196]	Time 2.029 (2.029)	Loss 0.7394 (0.7394)	Top 1-err 20.3125 (20.3125)	Top 5-err 5.0781 (5.0781)
* Epoch: [40/100]	 Top 1-err 32.922  Top 5-err 11.868	 Test Loss 1.321
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 41/100][Batch 0/5005]	 Loss 2.499, Top 1-error 33.984, Top 5-error 16.406
  Train with distillation: [Epoch 41/100][Batch 500/5005]	 Loss 2.447, Top 1-error 35.436, Top 5-error 15.005
  Train with distillation: [Epoch 41/100][Batch 1000/5005]	 Loss 2.452, Top 1-error 35.503, Top 5-error 15.113
  Train with distillation: [Epoch 41/100][Batch 1500/5005]	 Loss 2.461, Top 1-error 35.757, Top 5-error 15.227
  Train with distillation: [Epoch 41/100][Batch 2000/5005]	 Loss 2.461, Top 1-error 35.735, Top 5-error 15.253
  Train with distillation: [Epoch 41/100][Batch 2500/5005]	 Loss 2.463, Top 1-error 35.776, Top 5-error 15.291
  Train with distillation: [Epoch 41/100][Batch 3000/5005]	 Loss 2.465, Top 1-error 35.807, Top 5-error 15.306
  Train with distillation: [Epoch 41/100][Batch 3500/5005]	 Loss 2.467, Top 1-error 35.828, Top 5-error 15.325
  Train with distillation: [Epoch 41/100][Batch 4000/5005]	 Loss 2.470, Top 1-error 35.878, Top 5-error 15.367
  Train with distillation: [Epoch 41/100][Batch 4500/5005]	 Loss 2.472, Top 1-error 35.898, Top 5-error 15.371
  Train with distillation: [Epoch 41/100][Batch 5000/5005]	 Loss 2.474, Top 1-error 35.939, Top 5-error 15.372
  Train 	 Time Taken: 3066.77 sec
  Test (on val set): [Epoch 41/100][Batch 0/196]	Time 1.924 (1.924)	Loss 0.7501 (0.7501)	Top 1-err 21.4844 (21.4844)	Top 5-err 4.6875 (4.6875)
* Epoch: [41/100]	 Top 1-err 32.666  Top 5-err 11.812	 Test Loss 1.319
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 42/100][Batch 0/5005]	 Loss 2.548, Top 1-error 38.281, Top 5-error 16.797
  Train with distillation: [Epoch 42/100][Batch 500/5005]	 Loss 2.454, Top 1-error 35.562, Top 5-error 15.171
  Train with distillation: [Epoch 42/100][Batch 1000/5005]	 Loss 2.459, Top 1-error 35.620, Top 5-error 15.234
  Train with distillation: [Epoch 42/100][Batch 1500/5005]	 Loss 2.464, Top 1-error 35.719, Top 5-error 15.315
  Train with distillation: [Epoch 42/100][Batch 2000/5005]	 Loss 2.465, Top 1-error 35.727, Top 5-error 15.310
  Train with distillation: [Epoch 42/100][Batch 2500/5005]	 Loss 2.466, Top 1-error 35.726, Top 5-error 15.300
  Train with distillation: [Epoch 42/100][Batch 3000/5005]	 Loss 2.468, Top 1-error 35.771, Top 5-error 15.304
  Train with distillation: [Epoch 42/100][Batch 3500/5005]	 Loss 2.470, Top 1-error 35.817, Top 5-error 15.317
  Train with distillation: [Epoch 42/100][Batch 4000/5005]	 Loss 2.471, Top 1-error 35.829, Top 5-error 15.323
  Train with distillation: [Epoch 42/100][Batch 4500/5005]	 Loss 2.474, Top 1-error 35.879, Top 5-error 15.348
  Train with distillation: [Epoch 42/100][Batch 5000/5005]	 Loss 2.477, Top 1-error 35.942, Top 5-error 15.385
  Train 	 Time Taken: 3079.73 sec
  Test (on val set): [Epoch 42/100][Batch 0/196]	Time 1.996 (1.996)	Loss 0.7914 (0.7914)	Top 1-err 21.0938 (21.0938)	Top 5-err 3.5156 (3.5156)
* Epoch: [42/100]	 Top 1-err 32.700  Top 5-err 11.794	 Test Loss 1.314
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 43/100][Batch 0/5005]	 Loss 2.320, Top 1-error 33.984, Top 5-error 13.672
  Train with distillation: [Epoch 43/100][Batch 500/5005]	 Loss 2.458, Top 1-error 35.420, Top 5-error 15.143
  Train with distillation: [Epoch 43/100][Batch 1000/5005]	 Loss 2.469, Top 1-error 35.673, Top 5-error 15.251
  Train with distillation: [Epoch 43/100][Batch 1500/5005]	 Loss 2.475, Top 1-error 35.779, Top 5-error 15.311
  Train with distillation: [Epoch 43/100][Batch 2000/5005]	 Loss 2.477, Top 1-error 35.868, Top 5-error 15.352
  Train with distillation: [Epoch 43/100][Batch 2500/5005]	 Loss 2.478, Top 1-error 35.884, Top 5-error 15.357
  Train with distillation: [Epoch 43/100][Batch 3000/5005]	 Loss 2.479, Top 1-error 35.895, Top 5-error 15.387
  Train with distillation: [Epoch 43/100][Batch 3500/5005]	 Loss 2.481, Top 1-error 35.921, Top 5-error 15.414
  Train with distillation: [Epoch 43/100][Batch 4000/5005]	 Loss 2.482, Top 1-error 35.943, Top 5-error 15.422
  Train with distillation: [Epoch 43/100][Batch 4500/5005]	 Loss 2.483, Top 1-error 35.965, Top 5-error 15.443
  Train with distillation: [Epoch 43/100][Batch 5000/5005]	 Loss 2.484, Top 1-error 35.989, Top 5-error 15.447
  Train 	 Time Taken: 3094.87 sec
  Test (on val set): [Epoch 43/100][Batch 0/196]	Time 1.934 (1.934)	Loss 0.7522 (0.7522)	Top 1-err 21.4844 (21.4844)	Top 5-err 4.6875 (4.6875)
* Epoch: [43/100]	 Top 1-err 32.740  Top 5-err 11.774	 Test Loss 1.315
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 44/100][Batch 0/5005]	 Loss 2.164, Top 1-error 30.859, Top 5-error 11.719
  Train with distillation: [Epoch 44/100][Batch 500/5005]	 Loss 2.456, Top 1-error 35.516, Top 5-error 15.073
  Train with distillation: [Epoch 44/100][Batch 1000/5005]	 Loss 2.464, Top 1-error 35.648, Top 5-error 15.170
  Train with distillation: [Epoch 44/100][Batch 1500/5005]	 Loss 2.469, Top 1-error 35.732, Top 5-error 15.227
  Train with distillation: [Epoch 44/100][Batch 2000/5005]	 Loss 2.473, Top 1-error 35.789, Top 5-error 15.274
  Train with distillation: [Epoch 44/100][Batch 2500/5005]	 Loss 2.476, Top 1-error 35.834, Top 5-error 15.320
  Train with distillation: [Epoch 44/100][Batch 3000/5005]	 Loss 2.480, Top 1-error 35.892, Top 5-error 15.358
  Train with distillation: [Epoch 44/100][Batch 3500/5005]	 Loss 2.483, Top 1-error 35.948, Top 5-error 15.392
  Train with distillation: [Epoch 44/100][Batch 4000/5005]	 Loss 2.486, Top 1-error 35.984, Top 5-error 15.427
  Train with distillation: [Epoch 44/100][Batch 4500/5005]	 Loss 2.487, Top 1-error 35.988, Top 5-error 15.437
  Train with distillation: [Epoch 44/100][Batch 5000/5005]	 Loss 2.489, Top 1-error 36.025, Top 5-error 15.470
  Train 	 Time Taken: 3109.35 sec
  Test (on val set): [Epoch 44/100][Batch 0/196]	Time 2.035 (2.035)	Loss 0.6866 (0.6866)	Top 1-err 18.3594 (18.3594)	Top 5-err 2.7344 (2.7344)
* Epoch: [44/100]	 Top 1-err 33.000  Top 5-err 11.936	 Test Loss 1.326
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 45/100][Batch 0/5005]	 Loss 2.619, Top 1-error 37.109, Top 5-error 17.188
  Train with distillation: [Epoch 45/100][Batch 500/5005]	 Loss 2.467, Top 1-error 35.559, Top 5-error 15.181
  Train with distillation: [Epoch 45/100][Batch 1000/5005]	 Loss 2.472, Top 1-error 35.651, Top 5-error 15.288
  Train with distillation: [Epoch 45/100][Batch 1500/5005]	 Loss 2.477, Top 1-error 35.765, Top 5-error 15.374
  Train with distillation: [Epoch 45/100][Batch 2000/5005]	 Loss 2.477, Top 1-error 35.761, Top 5-error 15.349
  Train with distillation: [Epoch 45/100][Batch 2500/5005]	 Loss 2.482, Top 1-error 35.856, Top 5-error 15.396
  Train with distillation: [Epoch 45/100][Batch 3000/5005]	 Loss 2.483, Top 1-error 35.867, Top 5-error 15.386
  Train with distillation: [Epoch 45/100][Batch 3500/5005]	 Loss 2.486, Top 1-error 35.890, Top 5-error 15.417
  Train with distillation: [Epoch 45/100][Batch 4000/5005]	 Loss 2.489, Top 1-error 35.966, Top 5-error 15.444
  Train with distillation: [Epoch 45/100][Batch 4500/5005]	 Loss 2.491, Top 1-error 36.002, Top 5-error 15.481
  Train with distillation: [Epoch 45/100][Batch 5000/5005]	 Loss 2.493, Top 1-error 36.040, Top 5-error 15.507
  Train 	 Time Taken: 3125.32 sec
  Test (on val set): [Epoch 45/100][Batch 0/196]	Time 1.960 (1.960)	Loss 0.7568 (0.7568)	Top 1-err 19.5312 (19.5312)	Top 5-err 4.2969 (4.2969)
* Epoch: [45/100]	 Top 1-err 33.102  Top 5-err 11.918	 Test Loss 1.328
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 46/100][Batch 0/5005]	 Loss 2.302, Top 1-error 33.594, Top 5-error 14.062
  Train with distillation: [Epoch 46/100][Batch 500/5005]	 Loss 2.473, Top 1-error 35.770, Top 5-error 15.250
  Train with distillation: [Epoch 46/100][Batch 1000/5005]	 Loss 2.475, Top 1-error 35.808, Top 5-error 15.306
  Train with distillation: [Epoch 46/100][Batch 1500/5005]	 Loss 2.481, Top 1-error 35.842, Top 5-error 15.368
  Train with distillation: [Epoch 46/100][Batch 2000/5005]	 Loss 2.482, Top 1-error 35.848, Top 5-error 15.371
  Train with distillation: [Epoch 46/100][Batch 2500/5005]	 Loss 2.485, Top 1-error 35.881, Top 5-error 15.394
  Train with distillation: [Epoch 46/100][Batch 3000/5005]	 Loss 2.487, Top 1-error 35.919, Top 5-error 15.418
  Train with distillation: [Epoch 46/100][Batch 3500/5005]	 Loss 2.491, Top 1-error 35.989, Top 5-error 15.445
  Train with distillation: [Epoch 46/100][Batch 4000/5005]	 Loss 2.493, Top 1-error 36.019, Top 5-error 15.471
  Train with distillation: [Epoch 46/100][Batch 4500/5005]	 Loss 2.496, Top 1-error 36.058, Top 5-error 15.483
  Train with distillation: [Epoch 46/100][Batch 5000/5005]	 Loss 2.497, Top 1-error 36.083, Top 5-error 15.484
  Train 	 Time Taken: 3138.63 sec
  Test (on val set): [Epoch 46/100][Batch 0/196]	Time 2.051 (2.051)	Loss 0.7259 (0.7259)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.5156 (3.5156)
* Epoch: [46/100]	 Top 1-err 33.604  Top 5-err 12.208	 Test Loss 1.347
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 47/100][Batch 0/5005]	 Loss 2.439, Top 1-error 33.984, Top 5-error 12.109
  Train with distillation: [Epoch 47/100][Batch 500/5005]	 Loss 2.481, Top 1-error 35.761, Top 5-error 15.181
  Train with distillation: [Epoch 47/100][Batch 1000/5005]	 Loss 2.484, Top 1-error 35.759, Top 5-error 15.248
  Train with distillation: [Epoch 47/100][Batch 1500/5005]	 Loss 2.484, Top 1-error 35.776, Top 5-error 15.270
  Train with distillation: [Epoch 47/100][Batch 2000/5005]	 Loss 2.486, Top 1-error 35.832, Top 5-error 15.299
  Train with distillation: [Epoch 47/100][Batch 2500/5005]	 Loss 2.488, Top 1-error 35.899, Top 5-error 15.349
  Train with distillation: [Epoch 47/100][Batch 3000/5005]	 Loss 2.490, Top 1-error 35.917, Top 5-error 15.375
  Train with distillation: [Epoch 47/100][Batch 3500/5005]	 Loss 2.493, Top 1-error 35.965, Top 5-error 15.415
  Train with distillation: [Epoch 47/100][Batch 4000/5005]	 Loss 2.495, Top 1-error 36.009, Top 5-error 15.448
  Train with distillation: [Epoch 47/100][Batch 4500/5005]	 Loss 2.497, Top 1-error 36.027, Top 5-error 15.477
  Train with distillation: [Epoch 47/100][Batch 5000/5005]	 Loss 2.498, Top 1-error 36.044, Top 5-error 15.488
  Train 	 Time Taken: 3146.11 sec
  Test (on val set): [Epoch 47/100][Batch 0/196]	Time 1.914 (1.914)	Loss 0.7586 (0.7586)	Top 1-err 19.9219 (19.9219)	Top 5-err 6.2500 (6.2500)
* Epoch: [47/100]	 Top 1-err 33.198  Top 5-err 12.080	 Test Loss 1.335
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 48/100][Batch 0/5005]	 Loss 2.434, Top 1-error 35.156, Top 5-error 15.234
  Train with distillation: [Epoch 48/100][Batch 500/5005]	 Loss 2.485, Top 1-error 35.768, Top 5-error 15.417
  Train with distillation: [Epoch 48/100][Batch 1000/5005]	 Loss 2.484, Top 1-error 35.764, Top 5-error 15.378
  Train with distillation: [Epoch 48/100][Batch 1500/5005]	 Loss 2.486, Top 1-error 35.803, Top 5-error 15.398
  Train with distillation: [Epoch 48/100][Batch 2000/5005]	 Loss 2.487, Top 1-error 35.827, Top 5-error 15.406
  Train with distillation: [Epoch 48/100][Batch 2500/5005]	 Loss 2.490, Top 1-error 35.876, Top 5-error 15.428
  Train with distillation: [Epoch 48/100][Batch 3000/5005]	 Loss 2.492, Top 1-error 35.917, Top 5-error 15.450
  Train with distillation: [Epoch 48/100][Batch 3500/5005]	 Loss 2.495, Top 1-error 35.958, Top 5-error 15.478
  Train with distillation: [Epoch 48/100][Batch 4000/5005]	 Loss 2.497, Top 1-error 35.993, Top 5-error 15.510
  Train with distillation: [Epoch 48/100][Batch 4500/5005]	 Loss 2.499, Top 1-error 36.031, Top 5-error 15.531
  Train with distillation: [Epoch 48/100][Batch 5000/5005]	 Loss 2.501, Top 1-error 36.065, Top 5-error 15.550
  Train 	 Time Taken: 3158.62 sec
  Test (on val set): [Epoch 48/100][Batch 0/196]	Time 2.050 (2.050)	Loss 0.7036 (0.7036)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.5156 (3.5156)
* Epoch: [48/100]	 Top 1-err 32.868  Top 5-err 11.816	 Test Loss 1.326
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 49/100][Batch 0/5005]	 Loss 2.326, Top 1-error 35.547, Top 5-error 13.281
  Train with distillation: [Epoch 49/100][Batch 500/5005]	 Loss 2.486, Top 1-error 35.910, Top 5-error 15.283
  Train with distillation: [Epoch 49/100][Batch 1000/5005]	 Loss 2.490, Top 1-error 35.889, Top 5-error 15.369
  Train with distillation: [Epoch 49/100][Batch 1500/5005]	 Loss 2.493, Top 1-error 35.989, Top 5-error 15.427
  Train with distillation: [Epoch 49/100][Batch 2000/5005]	 Loss 2.494, Top 1-error 36.020, Top 5-error 15.471
  Train with distillation: [Epoch 49/100][Batch 2500/5005]	 Loss 2.497, Top 1-error 36.034, Top 5-error 15.469
  Train with distillation: [Epoch 49/100][Batch 3000/5005]	 Loss 2.497, Top 1-error 36.046, Top 5-error 15.473
  Train with distillation: [Epoch 49/100][Batch 3500/5005]	 Loss 2.499, Top 1-error 36.065, Top 5-error 15.498
  Train with distillation: [Epoch 49/100][Batch 4000/5005]	 Loss 2.500, Top 1-error 36.098, Top 5-error 15.496
  Train with distillation: [Epoch 49/100][Batch 4500/5005]	 Loss 2.502, Top 1-error 36.107, Top 5-error 15.511
  Train with distillation: [Epoch 49/100][Batch 5000/5005]	 Loss 2.502, Top 1-error 36.112, Top 5-error 15.507
  Train 	 Time Taken: 3160.61 sec
  Test (on val set): [Epoch 49/100][Batch 0/196]	Time 1.932 (1.932)	Loss 0.7402 (0.7402)	Top 1-err 20.7031 (20.7031)	Top 5-err 3.9062 (3.9062)
* Epoch: [49/100]	 Top 1-err 33.244  Top 5-err 12.168	 Test Loss 1.342
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 50/100][Batch 0/5005]	 Loss 2.635, Top 1-error 37.891, Top 5-error 16.406
  Train with distillation: [Epoch 50/100][Batch 500/5005]	 Loss 2.475, Top 1-error 35.597, Top 5-error 15.191
  Train with distillation: [Epoch 50/100][Batch 1000/5005]	 Loss 2.480, Top 1-error 35.703, Top 5-error 15.250
  Train with distillation: [Epoch 50/100][Batch 1500/5005]	 Loss 2.484, Top 1-error 35.722, Top 5-error 15.295
  Train with distillation: [Epoch 50/100][Batch 2000/5005]	 Loss 2.489, Top 1-error 35.817, Top 5-error 15.364
  Train with distillation: [Epoch 50/100][Batch 2500/5005]	 Loss 2.492, Top 1-error 35.871, Top 5-error 15.392
  Train with distillation: [Epoch 50/100][Batch 3000/5005]	 Loss 2.494, Top 1-error 35.893, Top 5-error 15.382
  Train with distillation: [Epoch 50/100][Batch 3500/5005]	 Loss 2.497, Top 1-error 35.963, Top 5-error 15.403
  Train with distillation: [Epoch 50/100][Batch 4000/5005]	 Loss 2.499, Top 1-error 36.005, Top 5-error 15.434
  Train with distillation: [Epoch 50/100][Batch 4500/5005]	 Loss 2.502, Top 1-error 36.061, Top 5-error 15.476
  Train with distillation: [Epoch 50/100][Batch 5000/5005]	 Loss 2.505, Top 1-error 36.116, Top 5-error 15.512
  Train 	 Time Taken: 3158.16 sec
  Test (on val set): [Epoch 50/100][Batch 0/196]	Time 2.054 (2.054)	Loss 0.7823 (0.7823)	Top 1-err 22.6562 (22.6562)	Top 5-err 5.4688 (5.4688)
* Epoch: [50/100]	 Top 1-err 33.030  Top 5-err 11.912	 Test Loss 1.335
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 51/100][Batch 0/5005]	 Loss 2.389, Top 1-error 33.203, Top 5-error 12.109
  Train with distillation: [Epoch 51/100][Batch 500/5005]	 Loss 2.485, Top 1-error 35.764, Top 5-error 15.284
  Train with distillation: [Epoch 51/100][Batch 1000/5005]	 Loss 2.486, Top 1-error 35.778, Top 5-error 15.307
  Train with distillation: [Epoch 51/100][Batch 1500/5005]	 Loss 2.487, Top 1-error 35.822, Top 5-error 15.299
  Train with distillation: [Epoch 51/100][Batch 2000/5005]	 Loss 2.492, Top 1-error 35.884, Top 5-error 15.330
  Train with distillation: [Epoch 51/100][Batch 2500/5005]	 Loss 2.494, Top 1-error 35.924, Top 5-error 15.356
  Train with distillation: [Epoch 51/100][Batch 3000/5005]	 Loss 2.498, Top 1-error 35.959, Top 5-error 15.437
  Train with distillation: [Epoch 51/100][Batch 3500/5005]	 Loss 2.501, Top 1-error 36.008, Top 5-error 15.468
  Train with distillation: [Epoch 51/100][Batch 4000/5005]	 Loss 2.502, Top 1-error 36.032, Top 5-error 15.478
  Train with distillation: [Epoch 51/100][Batch 4500/5005]	 Loss 2.503, Top 1-error 36.041, Top 5-error 15.497
  Train with distillation: [Epoch 51/100][Batch 5000/5005]	 Loss 2.505, Top 1-error 36.083, Top 5-error 15.527
  Train 	 Time Taken: 3141.87 sec
  Test (on val set): [Epoch 51/100][Batch 0/196]	Time 1.975 (1.975)	Loss 0.7931 (0.7931)	Top 1-err 23.0469 (23.0469)	Top 5-err 4.2969 (4.2969)
* Epoch: [51/100]	 Top 1-err 33.202  Top 5-err 11.956	 Test Loss 1.338
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 52/100][Batch 0/5005]	 Loss 2.574, Top 1-error 32.031, Top 5-error 17.969
  Train with distillation: [Epoch 52/100][Batch 500/5005]	 Loss 2.489, Top 1-error 35.759, Top 5-error 15.282
  Train with distillation: [Epoch 52/100][Batch 1000/5005]	 Loss 2.489, Top 1-error 35.740, Top 5-error 15.298
  Train with distillation: [Epoch 52/100][Batch 1500/5005]	 Loss 2.493, Top 1-error 35.792, Top 5-error 15.329
  Train with distillation: [Epoch 52/100][Batch 2000/5005]	 Loss 2.496, Top 1-error 35.863, Top 5-error 15.376
  Train with distillation: [Epoch 52/100][Batch 2500/5005]	 Loss 2.497, Top 1-error 35.881, Top 5-error 15.382
  Train with distillation: [Epoch 52/100][Batch 3000/5005]	 Loss 2.500, Top 1-error 35.916, Top 5-error 15.434
  Train with distillation: [Epoch 52/100][Batch 3500/5005]	 Loss 2.502, Top 1-error 35.984, Top 5-error 15.466
  Train with distillation: [Epoch 52/100][Batch 4000/5005]	 Loss 2.502, Top 1-error 35.976, Top 5-error 15.465
  Train with distillation: [Epoch 52/100][Batch 4500/5005]	 Loss 2.504, Top 1-error 36.020, Top 5-error 15.480
  Train with distillation: [Epoch 52/100][Batch 5000/5005]	 Loss 2.505, Top 1-error 36.055, Top 5-error 15.497
  Train 	 Time Taken: 3128.41 sec
  Test (on val set): [Epoch 52/100][Batch 0/196]	Time 2.049 (2.049)	Loss 0.8128 (0.8128)	Top 1-err 22.6562 (22.6562)	Top 5-err 5.0781 (5.0781)
* Epoch: [52/100]	 Top 1-err 33.172  Top 5-err 11.906	 Test Loss 1.331
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 53/100][Batch 0/5005]	 Loss 2.697, Top 1-error 39.453, Top 5-error 18.359
  Train with distillation: [Epoch 53/100][Batch 500/5005]	 Loss 2.485, Top 1-error 35.722, Top 5-error 15.287
  Train with distillation: [Epoch 53/100][Batch 1000/5005]	 Loss 2.488, Top 1-error 35.824, Top 5-error 15.325
  Train with distillation: [Epoch 53/100][Batch 1500/5005]	 Loss 2.488, Top 1-error 35.747, Top 5-error 15.350
  Train with distillation: [Epoch 53/100][Batch 2000/5005]	 Loss 2.492, Top 1-error 35.844, Top 5-error 15.362
  Train with distillation: [Epoch 53/100][Batch 2500/5005]	 Loss 2.498, Top 1-error 35.932, Top 5-error 15.416
  Train with distillation: [Epoch 53/100][Batch 3000/5005]	 Loss 2.500, Top 1-error 35.962, Top 5-error 15.428
  Train with distillation: [Epoch 53/100][Batch 3500/5005]	 Loss 2.503, Top 1-error 36.005, Top 5-error 15.473
  Train with distillation: [Epoch 53/100][Batch 4000/5005]	 Loss 2.504, Top 1-error 36.039, Top 5-error 15.487
  Train with distillation: [Epoch 53/100][Batch 4500/5005]	 Loss 2.505, Top 1-error 36.066, Top 5-error 15.483
  Train with distillation: [Epoch 53/100][Batch 5000/5005]	 Loss 2.508, Top 1-error 36.130, Top 5-error 15.532
  Train 	 Time Taken: 3132.33 sec
  Test (on val set): [Epoch 53/100][Batch 0/196]	Time 1.931 (1.931)	Loss 0.7607 (0.7607)	Top 1-err 22.6562 (22.6562)	Top 5-err 3.5156 (3.5156)
* Epoch: [53/100]	 Top 1-err 33.212  Top 5-err 11.986	 Test Loss 1.336
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 54/100][Batch 0/5005]	 Loss 2.246, Top 1-error 31.250, Top 5-error 11.328
  Train with distillation: [Epoch 54/100][Batch 500/5005]	 Loss 2.477, Top 1-error 35.654, Top 5-error 15.205
  Train with distillation: [Epoch 54/100][Batch 1000/5005]	 Loss 2.489, Top 1-error 35.907, Top 5-error 15.336
  Train with distillation: [Epoch 54/100][Batch 1500/5005]	 Loss 2.494, Top 1-error 36.008, Top 5-error 15.419
  Train with distillation: [Epoch 54/100][Batch 2000/5005]	 Loss 2.498, Top 1-error 36.044, Top 5-error 15.458
  Train with distillation: [Epoch 54/100][Batch 2500/5005]	 Loss 2.501, Top 1-error 36.076, Top 5-error 15.470
  Train with distillation: [Epoch 54/100][Batch 3000/5005]	 Loss 2.501, Top 1-error 36.066, Top 5-error 15.476
  Train with distillation: [Epoch 54/100][Batch 3500/5005]	 Loss 2.503, Top 1-error 36.087, Top 5-error 15.483
  Train with distillation: [Epoch 54/100][Batch 4000/5005]	 Loss 2.504, Top 1-error 36.102, Top 5-error 15.498
  Train with distillation: [Epoch 54/100][Batch 4500/5005]	 Loss 2.505, Top 1-error 36.114, Top 5-error 15.512
  Train with distillation: [Epoch 54/100][Batch 5000/5005]	 Loss 2.506, Top 1-error 36.125, Top 5-error 15.522
  Train 	 Time Taken: 3127.53 sec
  Test (on val set): [Epoch 54/100][Batch 0/196]	Time 2.004 (2.004)	Loss 0.7003 (0.7003)	Top 1-err 19.5312 (19.5312)	Top 5-err 3.1250 (3.1250)
* Epoch: [54/100]	 Top 1-err 33.636  Top 5-err 12.226	 Test Loss 1.346
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 55/100][Batch 0/5005]	 Loss 2.509, Top 1-error 35.547, Top 5-error 14.453
  Train with distillation: [Epoch 55/100][Batch 500/5005]	 Loss 2.478, Top 1-error 35.696, Top 5-error 15.220
  Train with distillation: [Epoch 55/100][Batch 1000/5005]	 Loss 2.484, Top 1-error 35.818, Top 5-error 15.226
  Train with distillation: [Epoch 55/100][Batch 1500/5005]	 Loss 2.487, Top 1-error 35.855, Top 5-error 15.306
  Train with distillation: [Epoch 55/100][Batch 2000/5005]	 Loss 2.492, Top 1-error 35.938, Top 5-error 15.354
  Train with distillation: [Epoch 55/100][Batch 2500/5005]	 Loss 2.496, Top 1-error 35.975, Top 5-error 15.379
  Train with distillation: [Epoch 55/100][Batch 3000/5005]	 Loss 2.496, Top 1-error 35.978, Top 5-error 15.384
  Train with distillation: [Epoch 55/100][Batch 3500/5005]	 Loss 2.499, Top 1-error 36.021, Top 5-error 15.425
  Train with distillation: [Epoch 55/100][Batch 4000/5005]	 Loss 2.502, Top 1-error 36.050, Top 5-error 15.460
  Train with distillation: [Epoch 55/100][Batch 4500/5005]	 Loss 2.502, Top 1-error 36.052, Top 5-error 15.474
  Train with distillation: [Epoch 55/100][Batch 5000/5005]	 Loss 2.503, Top 1-error 36.070, Top 5-error 15.485
  Train 	 Time Taken: 3124.34 sec
  Test (on val set): [Epoch 55/100][Batch 0/196]	Time 1.915 (1.915)	Loss 0.7044 (0.7044)	Top 1-err 21.0938 (21.0938)	Top 5-err 3.9062 (3.9062)
* Epoch: [55/100]	 Top 1-err 32.846  Top 5-err 11.864	 Test Loss 1.324
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 56/100][Batch 0/5005]	 Loss 2.476, Top 1-error 37.891, Top 5-error 12.500
  Train with distillation: [Epoch 56/100][Batch 500/5005]	 Loss 2.479, Top 1-error 35.602, Top 5-error 15.149
  Train with distillation: [Epoch 56/100][Batch 1000/5005]	 Loss 2.485, Top 1-error 35.775, Top 5-error 15.266
  Train with distillation: [Epoch 56/100][Batch 1500/5005]	 Loss 2.491, Top 1-error 35.889, Top 5-error 15.294
  Train with distillation: [Epoch 56/100][Batch 2000/5005]	 Loss 2.494, Top 1-error 35.901, Top 5-error 15.356
  Train with distillation: [Epoch 56/100][Batch 2500/5005]	 Loss 2.497, Top 1-error 35.934, Top 5-error 15.384
  Train with distillation: [Epoch 56/100][Batch 3000/5005]	 Loss 2.500, Top 1-error 35.981, Top 5-error 15.400
  Train with distillation: [Epoch 56/100][Batch 3500/5005]	 Loss 2.503, Top 1-error 36.064, Top 5-error 15.437
  Train with distillation: [Epoch 56/100][Batch 4000/5005]	 Loss 2.503, Top 1-error 36.079, Top 5-error 15.449
  Train with distillation: [Epoch 56/100][Batch 4500/5005]	 Loss 2.505, Top 1-error 36.091, Top 5-error 15.474
  Train with distillation: [Epoch 56/100][Batch 5000/5005]	 Loss 2.505, Top 1-error 36.091, Top 5-error 15.486
  Train 	 Time Taken: 3122.58 sec
  Test (on val set): [Epoch 56/100][Batch 0/196]	Time 2.038 (2.038)	Loss 0.7880 (0.7880)	Top 1-err 21.4844 (21.4844)	Top 5-err 3.9062 (3.9062)
* Epoch: [56/100]	 Top 1-err 32.848  Top 5-err 11.692	 Test Loss 1.317
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 57/100][Batch 0/5005]	 Loss 2.462, Top 1-error 37.891, Top 5-error 15.625
  Train with distillation: [Epoch 57/100][Batch 500/5005]	 Loss 2.474, Top 1-error 35.559, Top 5-error 15.044
  Train with distillation: [Epoch 57/100][Batch 1000/5005]	 Loss 2.478, Top 1-error 35.658, Top 5-error 15.130
  Train with distillation: [Epoch 57/100][Batch 1500/5005]	 Loss 2.481, Top 1-error 35.738, Top 5-error 15.194
  Train with distillation: [Epoch 57/100][Batch 2000/5005]	 Loss 2.486, Top 1-error 35.831, Top 5-error 15.256
  Train with distillation: [Epoch 57/100][Batch 2500/5005]	 Loss 2.489, Top 1-error 35.868, Top 5-error 15.322
  Train with distillation: [Epoch 57/100][Batch 3000/5005]	 Loss 2.490, Top 1-error 35.861, Top 5-error 15.338
  Train with distillation: [Epoch 57/100][Batch 3500/5005]	 Loss 2.493, Top 1-error 35.907, Top 5-error 15.383
  Train with distillation: [Epoch 57/100][Batch 4000/5005]	 Loss 2.495, Top 1-error 35.926, Top 5-error 15.401
  Train with distillation: [Epoch 57/100][Batch 4500/5005]	 Loss 2.495, Top 1-error 35.931, Top 5-error 15.402
  Train with distillation: [Epoch 57/100][Batch 5000/5005]	 Loss 2.497, Top 1-error 35.967, Top 5-error 15.418
  Train 	 Time Taken: 3118.80 sec
  Test (on val set): [Epoch 57/100][Batch 0/196]	Time 1.934 (1.934)	Loss 0.7439 (0.7439)	Top 1-err 19.9219 (19.9219)	Top 5-err 4.2969 (4.2969)
* Epoch: [57/100]	 Top 1-err 33.150  Top 5-err 11.812	 Test Loss 1.325
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 58/100][Batch 0/5005]	 Loss 2.368, Top 1-error 30.078, Top 5-error 12.891
  Train with distillation: [Epoch 58/100][Batch 500/5005]	 Loss 2.482, Top 1-error 35.566, Top 5-error 15.266
  Train with distillation: [Epoch 58/100][Batch 1000/5005]	 Loss 2.486, Top 1-error 35.644, Top 5-error 15.321
  Train with distillation: [Epoch 58/100][Batch 1500/5005]	 Loss 2.488, Top 1-error 35.779, Top 5-error 15.334
  Train with distillation: [Epoch 58/100][Batch 2000/5005]	 Loss 2.489, Top 1-error 35.799, Top 5-error 15.330
  Train with distillation: [Epoch 58/100][Batch 2500/5005]	 Loss 2.491, Top 1-error 35.842, Top 5-error 15.338
  Train with distillation: [Epoch 58/100][Batch 3000/5005]	 Loss 2.493, Top 1-error 35.831, Top 5-error 15.350
  Train with distillation: [Epoch 58/100][Batch 3500/5005]	 Loss 2.494, Top 1-error 35.873, Top 5-error 15.358
  Train with distillation: [Epoch 58/100][Batch 4000/5005]	 Loss 2.496, Top 1-error 35.906, Top 5-error 15.378
  Train with distillation: [Epoch 58/100][Batch 4500/5005]	 Loss 2.497, Top 1-error 35.939, Top 5-error 15.402
  Train with distillation: [Epoch 58/100][Batch 5000/5005]	 Loss 2.500, Top 1-error 35.983, Top 5-error 15.448
  Train 	 Time Taken: 3117.65 sec
  Test (on val set): [Epoch 58/100][Batch 0/196]	Time 2.009 (2.009)	Loss 0.7237 (0.7237)	Top 1-err 22.6562 (22.6562)	Top 5-err 3.1250 (3.1250)
* Epoch: [58/100]	 Top 1-err 33.002  Top 5-err 11.944	 Test Loss 1.330
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 59/100][Batch 0/5005]	 Loss 2.494, Top 1-error 36.328, Top 5-error 14.453
  Train with distillation: [Epoch 59/100][Batch 500/5005]	 Loss 2.475, Top 1-error 35.414, Top 5-error 15.187
  Train with distillation: [Epoch 59/100][Batch 1000/5005]	 Loss 2.481, Top 1-error 35.663, Top 5-error 15.231
  Train with distillation: [Epoch 59/100][Batch 1500/5005]	 Loss 2.487, Top 1-error 35.794, Top 5-error 15.284
  Train with distillation: [Epoch 59/100][Batch 2000/5005]	 Loss 2.490, Top 1-error 35.856, Top 5-error 15.307
  Train with distillation: [Epoch 59/100][Batch 2500/5005]	 Loss 2.493, Top 1-error 35.904, Top 5-error 15.338
  Train with distillation: [Epoch 59/100][Batch 3000/5005]	 Loss 2.495, Top 1-error 35.937, Top 5-error 15.378
  Train with distillation: [Epoch 59/100][Batch 3500/5005]	 Loss 2.495, Top 1-error 35.935, Top 5-error 15.382
  Train with distillation: [Epoch 59/100][Batch 4000/5005]	 Loss 2.498, Top 1-error 35.987, Top 5-error 15.433
  Train with distillation: [Epoch 59/100][Batch 4500/5005]	 Loss 2.500, Top 1-error 36.014, Top 5-error 15.436
  Train with distillation: [Epoch 59/100][Batch 5000/5005]	 Loss 2.502, Top 1-error 36.033, Top 5-error 15.463
  Train 	 Time Taken: 3114.12 sec
  Test (on val set): [Epoch 59/100][Batch 0/196]	Time 1.939 (1.939)	Loss 0.7456 (0.7456)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.9062 (3.9062)
* Epoch: [59/100]	 Top 1-err 33.270  Top 5-err 12.040	 Test Loss 1.338
  Current best accuracy (top-1 and 5 error): 32.428 11.576
  Train with distillation: [Epoch 60/100][Batch 0/5005]	 Loss 2.548, Top 1-error 40.234, Top 5-error 14.453
  Train with distillation: [Epoch 60/100][Batch 500/5005]	 Loss 2.332, Top 1-error 34.171, Top 5-error 14.384
  Train with distillation: [Epoch 60/100][Batch 1000/5005]	 Loss 2.304, Top 1-error 33.860, Top 5-error 14.230
  Train with distillation: [Epoch 60/100][Batch 1500/5005]	 Loss 2.288, Top 1-error 33.678, Top 5-error 14.094
  Train with distillation: [Epoch 60/100][Batch 2000/5005]	 Loss 2.275, Top 1-error 33.529, Top 5-error 14.003
  Train with distillation: [Epoch 60/100][Batch 2500/5005]	 Loss 2.268, Top 1-error 33.487, Top 5-error 13.974
  Train with distillation: [Epoch 60/100][Batch 3000/5005]	 Loss 2.260, Top 1-error 33.367, Top 5-error 13.916
  Train with distillation: [Epoch 60/100][Batch 3500/5005]	 Loss 2.254, Top 1-error 33.320, Top 5-error 13.872
  Train with distillation: [Epoch 60/100][Batch 4000/5005]	 Loss 2.249, Top 1-error 33.274, Top 5-error 13.838
  Train with distillation: [Epoch 60/100][Batch 4500/5005]	 Loss 2.244, Top 1-error 33.221, Top 5-error 13.798
  Train with distillation: [Epoch 60/100][Batch 5000/5005]	 Loss 2.239, Top 1-error 33.163, Top 5-error 13.756
  Train 	 Time Taken: 3110.50 sec
  Test (on val set): [Epoch 60/100][Batch 0/196]	Time 2.012 (2.012)	Loss 0.6460 (0.6460)	Top 1-err 18.3594 (18.3594)	Top 5-err 4.2969 (4.2969)
* Epoch: [60/100]	 Top 1-err 29.962  Top 5-err 10.224	 Test Loss 1.196
  Current best accuracy (top-1 and 5 error): 29.962 10.224
  Train with distillation: [Epoch 61/100][Batch 0/5005]	 Loss 2.408, Top 1-error 32.812, Top 5-error 16.016
  Train with distillation: [Epoch 61/100][Batch 500/5005]	 Loss 2.201, Top 1-error 32.695, Top 5-error 13.520
  Train with distillation: [Epoch 61/100][Batch 1000/5005]	 Loss 2.195, Top 1-error 32.639, Top 5-error 13.466
  Train with distillation: [Epoch 61/100][Batch 1500/5005]	 Loss 2.192, Top 1-error 32.596, Top 5-error 13.398
  Train with distillation: [Epoch 61/100][Batch 2000/5005]	 Loss 2.191, Top 1-error 32.586, Top 5-error 13.410
  Train with distillation: [Epoch 61/100][Batch 2500/5005]	 Loss 2.190, Top 1-error 32.596, Top 5-error 13.416
  Train with distillation: [Epoch 61/100][Batch 3000/5005]	 Loss 2.189, Top 1-error 32.548, Top 5-error 13.401
  Train with distillation: [Epoch 61/100][Batch 3500/5005]	 Loss 2.187, Top 1-error 32.546, Top 5-error 13.395
  Train with distillation: [Epoch 61/100][Batch 4000/5005]	 Loss 2.186, Top 1-error 32.532, Top 5-error 13.374
  Train with distillation: [Epoch 61/100][Batch 4500/5005]	 Loss 2.185, Top 1-error 32.520, Top 5-error 13.371
  Train with distillation: [Epoch 61/100][Batch 5000/5005]	 Loss 2.183, Top 1-error 32.515, Top 5-error 13.358
  Train 	 Time Taken: 3102.66 sec
  Test (on val set): [Epoch 61/100][Batch 0/196]	Time 1.873 (1.873)	Loss 0.6179 (0.6179)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.5156 (3.5156)
* Epoch: [61/100]	 Top 1-err 29.764  Top 5-err 10.180	 Test Loss 1.187
  Current best accuracy (top-1 and 5 error): 29.764 10.18
  Train with distillation: [Epoch 62/100][Batch 0/5005]	 Loss 2.199, Top 1-error 33.984, Top 5-error 14.062
  Train with distillation: [Epoch 62/100][Batch 500/5005]	 Loss 2.159, Top 1-error 32.248, Top 5-error 13.161
  Train with distillation: [Epoch 62/100][Batch 1000/5005]	 Loss 2.165, Top 1-error 32.339, Top 5-error 13.276
  Train with distillation: [Epoch 62/100][Batch 1500/5005]	 Loss 2.168, Top 1-error 32.363, Top 5-error 13.302
  Train with distillation: [Epoch 62/100][Batch 2000/5005]	 Loss 2.166, Top 1-error 32.349, Top 5-error 13.278
  Train with distillation: [Epoch 62/100][Batch 2500/5005]	 Loss 2.165, Top 1-error 32.314, Top 5-error 13.271
  Train with distillation: [Epoch 62/100][Batch 3000/5005]	 Loss 2.163, Top 1-error 32.266, Top 5-error 13.250
  Train with distillation: [Epoch 62/100][Batch 3500/5005]	 Loss 2.164, Top 1-error 32.278, Top 5-error 13.260
  Train with distillation: [Epoch 62/100][Batch 4000/5005]	 Loss 2.163, Top 1-error 32.256, Top 5-error 13.249
  Train with distillation: [Epoch 62/100][Batch 4500/5005]	 Loss 2.163, Top 1-error 32.267, Top 5-error 13.248
  Train with distillation: [Epoch 62/100][Batch 5000/5005]	 Loss 2.163, Top 1-error 32.262, Top 5-error 13.232
  Train 	 Time Taken: 3098.99 sec
  Test (on val set): [Epoch 62/100][Batch 0/196]	Time 2.071 (2.071)	Loss 0.6454 (0.6454)	Top 1-err 17.9688 (17.9688)	Top 5-err 4.6875 (4.6875)
* Epoch: [62/100]	 Top 1-err 29.612  Top 5-err 10.066	 Test Loss 1.180
  Current best accuracy (top-1 and 5 error): 29.612 10.066
  Train with distillation: [Epoch 63/100][Batch 0/5005]	 Loss 2.276, Top 1-error 34.766, Top 5-error 14.062
  Train with distillation: [Epoch 63/100][Batch 500/5005]	 Loss 2.142, Top 1-error 32.165, Top 5-error 13.004
  Train with distillation: [Epoch 63/100][Batch 1000/5005]	 Loss 2.144, Top 1-error 32.091, Top 5-error 13.068
  Train with distillation: [Epoch 63/100][Batch 1500/5005]	 Loss 2.147, Top 1-error 32.073, Top 5-error 13.099
  Train with distillation: [Epoch 63/100][Batch 2000/5005]	 Loss 2.147, Top 1-error 32.040, Top 5-error 13.092
  Train with distillation: [Epoch 63/100][Batch 2500/5005]	 Loss 2.147, Top 1-error 32.029, Top 5-error 13.105
  Train with distillation: [Epoch 63/100][Batch 3000/5005]	 Loss 2.149, Top 1-error 32.055, Top 5-error 13.115
  Train with distillation: [Epoch 63/100][Batch 3500/5005]	 Loss 2.149, Top 1-error 32.057, Top 5-error 13.126
  Train with distillation: [Epoch 63/100][Batch 4000/5005]	 Loss 2.149, Top 1-error 32.077, Top 5-error 13.154
  Train with distillation: [Epoch 63/100][Batch 4500/5005]	 Loss 2.148, Top 1-error 32.079, Top 5-error 13.146
  Train with distillation: [Epoch 63/100][Batch 5000/5005]	 Loss 2.149, Top 1-error 32.101, Top 5-error 13.156
  Train 	 Time Taken: 3098.23 sec
  Test (on val set): [Epoch 63/100][Batch 0/196]	Time 1.955 (1.955)	Loss 0.6215 (0.6215)	Top 1-err 17.5781 (17.5781)	Top 5-err 4.2969 (4.2969)
* Epoch: [63/100]	 Top 1-err 29.536  Top 5-err 9.982	 Test Loss 1.175
  Current best accuracy (top-1 and 5 error): 29.536 9.982
  Train with distillation: [Epoch 64/100][Batch 0/5005]	 Loss 2.151, Top 1-error 32.031, Top 5-error 13.672
  Train with distillation: [Epoch 64/100][Batch 500/5005]	 Loss 2.142, Top 1-error 32.027, Top 5-error 12.981
  Train with distillation: [Epoch 64/100][Batch 1000/5005]	 Loss 2.142, Top 1-error 32.017, Top 5-error 13.039
  Train with distillation: [Epoch 64/100][Batch 1500/5005]	 Loss 2.141, Top 1-error 32.002, Top 5-error 13.045
  Train with distillation: [Epoch 64/100][Batch 2000/5005]	 Loss 2.142, Top 1-error 31.987, Top 5-error 13.070
  Train with distillation: [Epoch 64/100][Batch 2500/5005]	 Loss 2.139, Top 1-error 31.955, Top 5-error 13.055
  Train with distillation: [Epoch 64/100][Batch 3000/5005]	 Loss 2.139, Top 1-error 31.937, Top 5-error 13.045
  Train with distillation: [Epoch 64/100][Batch 3500/5005]	 Loss 2.141, Top 1-error 31.994, Top 5-error 13.067
  Train with distillation: [Epoch 64/100][Batch 4000/5005]	 Loss 2.140, Top 1-error 31.963, Top 5-error 13.063
  Train with distillation: [Epoch 64/100][Batch 4500/5005]	 Loss 2.141, Top 1-error 31.957, Top 5-error 13.076
  Train with distillation: [Epoch 64/100][Batch 5000/5005]	 Loss 2.141, Top 1-error 31.971, Top 5-error 13.097
  Train 	 Time Taken: 3091.27 sec
  Test (on val set): [Epoch 64/100][Batch 0/196]	Time 2.037 (2.037)	Loss 0.6200 (0.6200)	Top 1-err 16.0156 (16.0156)	Top 5-err 4.6875 (4.6875)
* Epoch: [64/100]	 Top 1-err 29.488  Top 5-err 9.966	 Test Loss 1.173
  Current best accuracy (top-1 and 5 error): 29.488 9.966
  Train with distillation: [Epoch 65/100][Batch 0/5005]	 Loss 2.292, Top 1-error 33.984, Top 5-error 16.797
  Train with distillation: [Epoch 65/100][Batch 500/5005]	 Loss 2.135, Top 1-error 31.811, Top 5-error 12.983
  Train with distillation: [Epoch 65/100][Batch 1000/5005]	 Loss 2.129, Top 1-error 31.754, Top 5-error 12.904
  Train with distillation: [Epoch 65/100][Batch 1500/5005]	 Loss 2.129, Top 1-error 31.729, Top 5-error 12.918
  Train with distillation: [Epoch 65/100][Batch 2000/5005]	 Loss 2.131, Top 1-error 31.714, Top 5-error 12.937
  Train with distillation: [Epoch 65/100][Batch 2500/5005]	 Loss 2.131, Top 1-error 31.744, Top 5-error 12.937
  Train with distillation: [Epoch 65/100][Batch 3000/5005]	 Loss 2.132, Top 1-error 31.791, Top 5-error 12.959
  Train with distillation: [Epoch 65/100][Batch 3500/5005]	 Loss 2.133, Top 1-error 31.808, Top 5-error 12.977
  Train with distillation: [Epoch 65/100][Batch 4000/5005]	 Loss 2.133, Top 1-error 31.806, Top 5-error 12.973
  Train with distillation: [Epoch 65/100][Batch 4500/5005]	 Loss 2.132, Top 1-error 31.811, Top 5-error 12.963
  Train with distillation: [Epoch 65/100][Batch 5000/5005]	 Loss 2.133, Top 1-error 31.821, Top 5-error 12.971
  Train 	 Time Taken: 3090.49 sec
  Test (on val set): [Epoch 65/100][Batch 0/196]	Time 1.966 (1.966)	Loss 0.6165 (0.6165)	Top 1-err 17.1875 (17.1875)	Top 5-err 3.1250 (3.1250)
* Epoch: [65/100]	 Top 1-err 29.262  Top 5-err 9.936	 Test Loss 1.171
  Current best accuracy (top-1 and 5 error): 29.262 9.936
  Train with distillation: [Epoch 66/100][Batch 0/5005]	 Loss 2.309, Top 1-error 37.109, Top 5-error 14.453
  Train with distillation: [Epoch 66/100][Batch 500/5005]	 Loss 2.123, Top 1-error 31.599, Top 5-error 12.935
  Train with distillation: [Epoch 66/100][Batch 1000/5005]	 Loss 2.124, Top 1-error 31.706, Top 5-error 12.938
  Train with distillation: [Epoch 66/100][Batch 1500/5005]	 Loss 2.128, Top 1-error 31.812, Top 5-error 12.995
  Train with distillation: [Epoch 66/100][Batch 2000/5005]	 Loss 2.129, Top 1-error 31.823, Top 5-error 13.006
  Train with distillation: [Epoch 66/100][Batch 2500/5005]	 Loss 2.128, Top 1-error 31.804, Top 5-error 13.001
  Train with distillation: [Epoch 66/100][Batch 3000/5005]	 Loss 2.126, Top 1-error 31.776, Top 5-error 12.969
  Train with distillation: [Epoch 66/100][Batch 3500/5005]	 Loss 2.125, Top 1-error 31.761, Top 5-error 12.934
  Train with distillation: [Epoch 66/100][Batch 4000/5005]	 Loss 2.126, Top 1-error 31.753, Top 5-error 12.945
  Train with distillation: [Epoch 66/100][Batch 4500/5005]	 Loss 2.125, Top 1-error 31.746, Top 5-error 12.931
  Train with distillation: [Epoch 66/100][Batch 5000/5005]	 Loss 2.125, Top 1-error 31.739, Top 5-error 12.927
  Train 	 Time Taken: 3088.28 sec
  Test (on val set): [Epoch 66/100][Batch 0/196]	Time 2.045 (2.045)	Loss 0.6330 (0.6330)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.5156 (3.5156)
* Epoch: [66/100]	 Top 1-err 29.332  Top 5-err 9.852	 Test Loss 1.170
  Current best accuracy (top-1 and 5 error): 29.262 9.936
  Train with distillation: [Epoch 67/100][Batch 0/5005]	 Loss 2.017, Top 1-error 31.250, Top 5-error 11.328
  Train with distillation: [Epoch 67/100][Batch 500/5005]	 Loss 2.117, Top 1-error 31.692, Top 5-error 12.890
  Train with distillation: [Epoch 67/100][Batch 1000/5005]	 Loss 2.119, Top 1-error 31.606, Top 5-error 12.882
  Train with distillation: [Epoch 67/100][Batch 1500/5005]	 Loss 2.119, Top 1-error 31.612, Top 5-error 12.898
  Train with distillation: [Epoch 67/100][Batch 2000/5005]	 Loss 2.119, Top 1-error 31.621, Top 5-error 12.894
  Train with distillation: [Epoch 67/100][Batch 2500/5005]	 Loss 2.120, Top 1-error 31.666, Top 5-error 12.903
  Train with distillation: [Epoch 67/100][Batch 3000/5005]	 Loss 2.119, Top 1-error 31.652, Top 5-error 12.878
  Train with distillation: [Epoch 67/100][Batch 3500/5005]	 Loss 2.118, Top 1-error 31.615, Top 5-error 12.852
  Train with distillation: [Epoch 67/100][Batch 4000/5005]	 Loss 2.119, Top 1-error 31.602, Top 5-error 12.866
  Train with distillation: [Epoch 67/100][Batch 4500/5005]	 Loss 2.119, Top 1-error 31.631, Top 5-error 12.879
  Train with distillation: [Epoch 67/100][Batch 5000/5005]	 Loss 2.119, Top 1-error 31.640, Top 5-error 12.883
  Train 	 Time Taken: 3090.40 sec
  Test (on val set): [Epoch 67/100][Batch 0/196]	Time 1.905 (1.905)	Loss 0.6204 (0.6204)	Top 1-err 17.1875 (17.1875)	Top 5-err 3.9062 (3.9062)
* Epoch: [67/100]	 Top 1-err 29.316  Top 5-err 9.900	 Test Loss 1.167
  Current best accuracy (top-1 and 5 error): 29.262 9.936
  Train with distillation: [Epoch 68/100][Batch 0/5005]	 Loss 2.068, Top 1-error 31.641, Top 5-error 11.328
  Train with distillation: [Epoch 68/100][Batch 500/5005]	 Loss 2.108, Top 1-error 31.515, Top 5-error 12.714
  Train with distillation: [Epoch 68/100][Batch 1000/5005]	 Loss 2.113, Top 1-error 31.574, Top 5-error 12.780
  Train with distillation: [Epoch 68/100][Batch 1500/5005]	 Loss 2.114, Top 1-error 31.558, Top 5-error 12.808
  Train with distillation: [Epoch 68/100][Batch 2000/5005]	 Loss 2.116, Top 1-error 31.605, Top 5-error 12.859
  Train with distillation: [Epoch 68/100][Batch 2500/5005]	 Loss 2.115, Top 1-error 31.607, Top 5-error 12.846
  Train with distillation: [Epoch 68/100][Batch 3000/5005]	 Loss 2.113, Top 1-error 31.593, Top 5-error 12.814
  Train with distillation: [Epoch 68/100][Batch 3500/5005]	 Loss 2.113, Top 1-error 31.600, Top 5-error 12.822
  Train with distillation: [Epoch 68/100][Batch 4000/5005]	 Loss 2.114, Top 1-error 31.620, Top 5-error 12.829
  Train with distillation: [Epoch 68/100][Batch 4500/5005]	 Loss 2.115, Top 1-error 31.605, Top 5-error 12.845
  Train with distillation: [Epoch 68/100][Batch 5000/5005]	 Loss 2.115, Top 1-error 31.605, Top 5-error 12.855
  Train 	 Time Taken: 3100.91 sec
  Test (on val set): [Epoch 68/100][Batch 0/196]	Time 1.995 (1.995)	Loss 0.6451 (0.6451)	Top 1-err 17.1875 (17.1875)	Top 5-err 3.9062 (3.9062)
* Epoch: [68/100]	 Top 1-err 29.266  Top 5-err 9.836	 Test Loss 1.165
  Current best accuracy (top-1 and 5 error): 29.262 9.936
  Train with distillation: [Epoch 69/100][Batch 0/5005]	 Loss 2.032, Top 1-error 33.594, Top 5-error 12.500
  Train with distillation: [Epoch 69/100][Batch 500/5005]	 Loss 2.112, Top 1-error 31.609, Top 5-error 12.878
  Train with distillation: [Epoch 69/100][Batch 1000/5005]	 Loss 2.105, Top 1-error 31.407, Top 5-error 12.736
  Train with distillation: [Epoch 69/100][Batch 1500/5005]	 Loss 2.105, Top 1-error 31.437, Top 5-error 12.732
  Train with distillation: [Epoch 69/100][Batch 2000/5005]	 Loss 2.106, Top 1-error 31.468, Top 5-error 12.731
  Train with distillation: [Epoch 69/100][Batch 2500/5005]	 Loss 2.107, Top 1-error 31.477, Top 5-error 12.754
  Train with distillation: [Epoch 69/100][Batch 3000/5005]	 Loss 2.108, Top 1-error 31.479, Top 5-error 12.764
  Train with distillation: [Epoch 69/100][Batch 3500/5005]	 Loss 2.107, Top 1-error 31.477, Top 5-error 12.758
  Train with distillation: [Epoch 69/100][Batch 4000/5005]	 Loss 2.107, Top 1-error 31.475, Top 5-error 12.761
  Train with distillation: [Epoch 69/100][Batch 4500/5005]	 Loss 2.108, Top 1-error 31.491, Top 5-error 12.781
  Train with distillation: [Epoch 69/100][Batch 5000/5005]	 Loss 2.109, Top 1-error 31.512, Top 5-error 12.794
  Train 	 Time Taken: 3117.54 sec
  Test (on val set): [Epoch 69/100][Batch 0/196]	Time 1.924 (1.924)	Loss 0.6221 (0.6221)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.5156 (3.5156)
* Epoch: [69/100]	 Top 1-err 29.224  Top 5-err 9.834	 Test Loss 1.161
  Current best accuracy (top-1 and 5 error): 29.224 9.834
  Train with distillation: [Epoch 70/100][Batch 0/5005]	 Loss 2.114, Top 1-error 32.031, Top 5-error 12.891
  Train with distillation: [Epoch 70/100][Batch 500/5005]	 Loss 2.107, Top 1-error 31.571, Top 5-error 12.820
  Train with distillation: [Epoch 70/100][Batch 1000/5005]	 Loss 2.105, Top 1-error 31.455, Top 5-error 12.791
  Train with distillation: [Epoch 70/100][Batch 1500/5005]	 Loss 2.103, Top 1-error 31.443, Top 5-error 12.742
  Train with distillation: [Epoch 70/100][Batch 2000/5005]	 Loss 2.104, Top 1-error 31.431, Top 5-error 12.770
  Train with distillation: [Epoch 70/100][Batch 2500/5005]	 Loss 2.104, Top 1-error 31.434, Top 5-error 12.774
  Train with distillation: [Epoch 70/100][Batch 3000/5005]	 Loss 2.106, Top 1-error 31.484, Top 5-error 12.795
  Train with distillation: [Epoch 70/100][Batch 3500/5005]	 Loss 2.106, Top 1-error 31.469, Top 5-error 12.778
  Train with distillation: [Epoch 70/100][Batch 4000/5005]	 Loss 2.107, Top 1-error 31.478, Top 5-error 12.796
  Train with distillation: [Epoch 70/100][Batch 4500/5005]	 Loss 2.106, Top 1-error 31.468, Top 5-error 12.774
  Train with distillation: [Epoch 70/100][Batch 5000/5005]	 Loss 2.106, Top 1-error 31.489, Top 5-error 12.784
  Train 	 Time Taken: 3136.37 sec
  Test (on val set): [Epoch 70/100][Batch 0/196]	Time 1.987 (1.987)	Loss 0.6376 (0.6376)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.5156 (3.5156)
* Epoch: [70/100]	 Top 1-err 29.226  Top 5-err 9.906	 Test Loss 1.161
  Current best accuracy (top-1 and 5 error): 29.224 9.834
  Train with distillation: [Epoch 71/100][Batch 0/5005]	 Loss 1.947, Top 1-error 28.516, Top 5-error 11.328
  Train with distillation: [Epoch 71/100][Batch 500/5005]	 Loss 2.095, Top 1-error 31.262, Top 5-error 12.662
  Train with distillation: [Epoch 71/100][Batch 1000/5005]	 Loss 2.092, Top 1-error 31.323, Top 5-error 12.578
  Train with distillation: [Epoch 71/100][Batch 1500/5005]	 Loss 2.093, Top 1-error 31.338, Top 5-error 12.593
  Train with distillation: [Epoch 71/100][Batch 2000/5005]	 Loss 2.096, Top 1-error 31.380, Top 5-error 12.630
  Train with distillation: [Epoch 71/100][Batch 2500/5005]	 Loss 2.097, Top 1-error 31.345, Top 5-error 12.649
  Train with distillation: [Epoch 71/100][Batch 3000/5005]	 Loss 2.097, Top 1-error 31.346, Top 5-error 12.646
  Train with distillation: [Epoch 71/100][Batch 3500/5005]	 Loss 2.098, Top 1-error 31.360, Top 5-error 12.649
  Train with distillation: [Epoch 71/100][Batch 4000/5005]	 Loss 2.098, Top 1-error 31.364, Top 5-error 12.661
  Train with distillation: [Epoch 71/100][Batch 4500/5005]	 Loss 2.099, Top 1-error 31.383, Top 5-error 12.673
  Train with distillation: [Epoch 71/100][Batch 5000/5005]	 Loss 2.099, Top 1-error 31.391, Top 5-error 12.677
  Train 	 Time Taken: 3157.12 sec
  Test (on val set): [Epoch 71/100][Batch 0/196]	Time 1.911 (1.911)	Loss 0.6158 (0.6158)	Top 1-err 16.4062 (16.4062)	Top 5-err 3.9062 (3.9062)
* Epoch: [71/100]	 Top 1-err 29.208  Top 5-err 9.858	 Test Loss 1.161
  Current best accuracy (top-1 and 5 error): 29.208 9.858
  Train with distillation: [Epoch 72/100][Batch 0/5005]	 Loss 2.102, Top 1-error 30.859, Top 5-error 12.891
  Train with distillation: [Epoch 72/100][Batch 500/5005]	 Loss 2.110, Top 1-error 31.599, Top 5-error 12.916
  Train with distillation: [Epoch 72/100][Batch 1000/5005]	 Loss 2.105, Top 1-error 31.458, Top 5-error 12.803
  Train with distillation: [Epoch 72/100][Batch 1500/5005]	 Loss 2.103, Top 1-error 31.390, Top 5-error 12.797
  Train with distillation: [Epoch 72/100][Batch 2000/5005]	 Loss 2.100, Top 1-error 31.338, Top 5-error 12.728
  Train with distillation: [Epoch 72/100][Batch 2500/5005]	 Loss 2.101, Top 1-error 31.389, Top 5-error 12.748
  Train with distillation: [Epoch 72/100][Batch 3000/5005]	 Loss 2.100, Top 1-error 31.376, Top 5-error 12.716
  Train with distillation: [Epoch 72/100][Batch 3500/5005]	 Loss 2.100, Top 1-error 31.381, Top 5-error 12.732
  Train with distillation: [Epoch 72/100][Batch 4000/5005]	 Loss 2.101, Top 1-error 31.407, Top 5-error 12.751
  Train with distillation: [Epoch 72/100][Batch 4500/5005]	 Loss 2.101, Top 1-error 31.407, Top 5-error 12.753
  Train with distillation: [Epoch 72/100][Batch 5000/5005]	 Loss 2.101, Top 1-error 31.415, Top 5-error 12.767
  Train 	 Time Taken: 3160.61 sec
  Test (on val set): [Epoch 72/100][Batch 0/196]	Time 1.996 (1.996)	Loss 0.6011 (0.6011)	Top 1-err 16.4062 (16.4062)	Top 5-err 4.6875 (4.6875)
* Epoch: [72/100]	 Top 1-err 29.042  Top 5-err 9.846	 Test Loss 1.157
  Current best accuracy (top-1 and 5 error): 29.042 9.846
  Train with distillation: [Epoch 73/100][Batch 0/5005]	 Loss 2.041, Top 1-error 31.250, Top 5-error 12.891
  Train with distillation: [Epoch 73/100][Batch 500/5005]	 Loss 2.081, Top 1-error 31.037, Top 5-error 12.354
  Train with distillation: [Epoch 73/100][Batch 1000/5005]	 Loss 2.092, Top 1-error 31.238, Top 5-error 12.540
  Train with distillation: [Epoch 73/100][Batch 1500/5005]	 Loss 2.093, Top 1-error 31.261, Top 5-error 12.585
  Train with distillation: [Epoch 73/100][Batch 2000/5005]	 Loss 2.092, Top 1-error 31.285, Top 5-error 12.606
  Train with distillation: [Epoch 73/100][Batch 2500/5005]	 Loss 2.091, Top 1-error 31.287, Top 5-error 12.597
  Train with distillation: [Epoch 73/100][Batch 3000/5005]	 Loss 2.092, Top 1-error 31.319, Top 5-error 12.596
  Train with distillation: [Epoch 73/100][Batch 3500/5005]	 Loss 2.093, Top 1-error 31.330, Top 5-error 12.625
  Train with distillation: [Epoch 73/100][Batch 4000/5005]	 Loss 2.094, Top 1-error 31.353, Top 5-error 12.635
  Train with distillation: [Epoch 73/100][Batch 4500/5005]	 Loss 2.094, Top 1-error 31.339, Top 5-error 12.633
  Train with distillation: [Epoch 73/100][Batch 5000/5005]	 Loss 2.094, Top 1-error 31.338, Top 5-error 12.633
  Train 	 Time Taken: 3179.20 sec
  Test (on val set): [Epoch 73/100][Batch 0/196]	Time 1.927 (1.927)	Loss 0.6123 (0.6123)	Top 1-err 16.4062 (16.4062)	Top 5-err 3.5156 (3.5156)
* Epoch: [73/100]	 Top 1-err 29.104  Top 5-err 9.834	 Test Loss 1.158
  Current best accuracy (top-1 and 5 error): 29.042 9.846
  Train with distillation: [Epoch 74/100][Batch 0/5005]	 Loss 1.882, Top 1-error 28.906, Top 5-error 8.984
  Train with distillation: [Epoch 74/100][Batch 500/5005]	 Loss 2.092, Top 1-error 31.245, Top 5-error 12.466
  Train with distillation: [Epoch 74/100][Batch 1000/5005]	 Loss 2.092, Top 1-error 31.213, Top 5-error 12.564
  Train with distillation: [Epoch 74/100][Batch 1500/5005]	 Loss 2.092, Top 1-error 31.166, Top 5-error 12.596
  Train with distillation: [Epoch 74/100][Batch 2000/5005]	 Loss 2.092, Top 1-error 31.204, Top 5-error 12.603
  Train with distillation: [Epoch 74/100][Batch 2500/5005]	 Loss 2.090, Top 1-error 31.184, Top 5-error 12.588
  Train with distillation: [Epoch 74/100][Batch 3000/5005]	 Loss 2.090, Top 1-error 31.205, Top 5-error 12.608
  Train with distillation: [Epoch 74/100][Batch 3500/5005]	 Loss 2.091, Top 1-error 31.211, Top 5-error 12.629
  Train with distillation: [Epoch 74/100][Batch 4000/5005]	 Loss 2.092, Top 1-error 31.231, Top 5-error 12.637
  Train with distillation: [Epoch 74/100][Batch 4500/5005]	 Loss 2.092, Top 1-error 31.244, Top 5-error 12.655
  Train with distillation: [Epoch 74/100][Batch 5000/5005]	 Loss 2.092, Top 1-error 31.232, Top 5-error 12.659
  Train 	 Time Taken: 3171.35 sec
  Test (on val set): [Epoch 74/100][Batch 0/196]	Time 2.021 (2.021)	Loss 0.6099 (0.6099)	Top 1-err 17.1875 (17.1875)	Top 5-err 3.5156 (3.5156)
* Epoch: [74/100]	 Top 1-err 29.106  Top 5-err 9.744	 Test Loss 1.155
  Current best accuracy (top-1 and 5 error): 29.042 9.846
  Train with distillation: [Epoch 75/100][Batch 0/5005]	 Loss 2.118, Top 1-error 30.469, Top 5-error 10.547
  Train with distillation: [Epoch 75/100][Batch 500/5005]	 Loss 2.088, Top 1-error 31.172, Top 5-error 12.707
  Train with distillation: [Epoch 75/100][Batch 1000/5005]	 Loss 2.090, Top 1-error 31.205, Top 5-error 12.715
  Train with distillation: [Epoch 75/100][Batch 1500/5005]	 Loss 2.087, Top 1-error 31.121, Top 5-error 12.636
  Train with distillation: [Epoch 75/100][Batch 2000/5005]	 Loss 2.088, Top 1-error 31.194, Top 5-error 12.624
  Train with distillation: [Epoch 75/100][Batch 2500/5005]	 Loss 2.088, Top 1-error 31.209, Top 5-error 12.634
  Train with distillation: [Epoch 75/100][Batch 3000/5005]	 Loss 2.087, Top 1-error 31.183, Top 5-error 12.610
  Train with distillation: [Epoch 75/100][Batch 3500/5005]	 Loss 2.088, Top 1-error 31.190, Top 5-error 12.608
  Train with distillation: [Epoch 75/100][Batch 4000/5005]	 Loss 2.088, Top 1-error 31.205, Top 5-error 12.603
  Train with distillation: [Epoch 75/100][Batch 4500/5005]	 Loss 2.090, Top 1-error 31.243, Top 5-error 12.616
  Train with distillation: [Epoch 75/100][Batch 5000/5005]	 Loss 2.090, Top 1-error 31.243, Top 5-error 12.633
  Train 	 Time Taken: 3184.21 sec
  Test (on val set): [Epoch 75/100][Batch 0/196]	Time 1.972 (1.972)	Loss 0.5775 (0.5775)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.5156 (3.5156)
* Epoch: [75/100]	 Top 1-err 29.096  Top 5-err 9.730	 Test Loss 1.156
  Current best accuracy (top-1 and 5 error): 29.042 9.846
  Train with distillation: [Epoch 76/100][Batch 0/5005]	 Loss 2.329, Top 1-error 35.156, Top 5-error 16.406
  Train with distillation: [Epoch 76/100][Batch 500/5005]	 Loss 2.086, Top 1-error 31.179, Top 5-error 12.579
  Train with distillation: [Epoch 76/100][Batch 1000/5005]	 Loss 2.082, Top 1-error 31.133, Top 5-error 12.487
  Train with distillation: [Epoch 76/100][Batch 1500/5005]	 Loss 2.081, Top 1-error 31.093, Top 5-error 12.518
  Train with distillation: [Epoch 76/100][Batch 2000/5005]	 Loss 2.083, Top 1-error 31.113, Top 5-error 12.567
  Train with distillation: [Epoch 76/100][Batch 2500/5005]	 Loss 2.084, Top 1-error 31.141, Top 5-error 12.569
  Train with distillation: [Epoch 76/100][Batch 3000/5005]	 Loss 2.085, Top 1-error 31.156, Top 5-error 12.581
  Train with distillation: [Epoch 76/100][Batch 3500/5005]	 Loss 2.085, Top 1-error 31.143, Top 5-error 12.582
  Train with distillation: [Epoch 76/100][Batch 4000/5005]	 Loss 2.086, Top 1-error 31.153, Top 5-error 12.575
  Train with distillation: [Epoch 76/100][Batch 4500/5005]	 Loss 2.085, Top 1-error 31.142, Top 5-error 12.557
  Train with distillation: [Epoch 76/100][Batch 5000/5005]	 Loss 2.085, Top 1-error 31.158, Top 5-error 12.559
  Train 	 Time Taken: 3187.39 sec
  Test (on val set): [Epoch 76/100][Batch 0/196]	Time 1.978 (1.978)	Loss 0.6106 (0.6106)	Top 1-err 16.0156 (16.0156)	Top 5-err 3.5156 (3.5156)
* Epoch: [76/100]	 Top 1-err 28.974  Top 5-err 9.670	 Test Loss 1.152
  Current best accuracy (top-1 and 5 error): 28.974 9.67
  Train with distillation: [Epoch 77/100][Batch 0/5005]	 Loss 2.107, Top 1-error 30.469, Top 5-error 13.281
  Train with distillation: [Epoch 77/100][Batch 500/5005]	 Loss 2.078, Top 1-error 30.919, Top 5-error 12.548
  Train with distillation: [Epoch 77/100][Batch 1000/5005]	 Loss 2.082, Top 1-error 31.000, Top 5-error 12.603
  Train with distillation: [Epoch 77/100][Batch 1500/5005]	 Loss 2.083, Top 1-error 31.091, Top 5-error 12.593
  Train with distillation: [Epoch 77/100][Batch 2000/5005]	 Loss 2.084, Top 1-error 31.101, Top 5-error 12.575
  Train with distillation: [Epoch 77/100][Batch 2500/5005]	 Loss 2.085, Top 1-error 31.104, Top 5-error 12.591
  Train with distillation: [Epoch 77/100][Batch 3000/5005]	 Loss 2.083, Top 1-error 31.059, Top 5-error 12.578
  Train with distillation: [Epoch 77/100][Batch 3500/5005]	 Loss 2.083, Top 1-error 31.071, Top 5-error 12.568
  Train with distillation: [Epoch 77/100][Batch 4000/5005]	 Loss 2.084, Top 1-error 31.090, Top 5-error 12.575
  Train with distillation: [Epoch 77/100][Batch 4500/5005]	 Loss 2.085, Top 1-error 31.110, Top 5-error 12.587
  Train with distillation: [Epoch 77/100][Batch 5000/5005]	 Loss 2.085, Top 1-error 31.118, Top 5-error 12.581
  Train 	 Time Taken: 3189.35 sec
  Test (on val set): [Epoch 77/100][Batch 0/196]	Time 1.947 (1.947)	Loss 0.6067 (0.6067)	Top 1-err 16.4062 (16.4062)	Top 5-err 3.5156 (3.5156)
* Epoch: [77/100]	 Top 1-err 29.014  Top 5-err 9.774	 Test Loss 1.152
  Current best accuracy (top-1 and 5 error): 28.974 9.67
  Train with distillation: [Epoch 78/100][Batch 0/5005]	 Loss 2.037, Top 1-error 30.078, Top 5-error 10.938
  Train with distillation: [Epoch 78/100][Batch 500/5005]	 Loss 2.087, Top 1-error 31.240, Top 5-error 12.652
  Train with distillation: [Epoch 78/100][Batch 1000/5005]	 Loss 2.084, Top 1-error 31.168, Top 5-error 12.592
  Train with distillation: [Epoch 78/100][Batch 1500/5005]	 Loss 2.083, Top 1-error 31.147, Top 5-error 12.545
  Train with distillation: [Epoch 78/100][Batch 2000/5005]	 Loss 2.085, Top 1-error 31.189, Top 5-error 12.572
  Train with distillation: [Epoch 78/100][Batch 2500/5005]	 Loss 2.085, Top 1-error 31.159, Top 5-error 12.563
  Train with distillation: [Epoch 78/100][Batch 3000/5005]	 Loss 2.084, Top 1-error 31.115, Top 5-error 12.543
  Train with distillation: [Epoch 78/100][Batch 3500/5005]	 Loss 2.085, Top 1-error 31.155, Top 5-error 12.568
  Train with distillation: [Epoch 78/100][Batch 4000/5005]	 Loss 2.084, Top 1-error 31.156, Top 5-error 12.560
  Train with distillation: [Epoch 78/100][Batch 4500/5005]	 Loss 2.083, Top 1-error 31.142, Top 5-error 12.558
  Train with distillation: [Epoch 78/100][Batch 5000/5005]	 Loss 2.084, Top 1-error 31.143, Top 5-error 12.560
  Train 	 Time Taken: 3179.08 sec
  Test (on val set): [Epoch 78/100][Batch 0/196]	Time 2.004 (2.004)	Loss 0.6403 (0.6403)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.9062 (3.9062)
* Epoch: [78/100]	 Top 1-err 29.090  Top 5-err 9.728	 Test Loss 1.152
  Current best accuracy (top-1 and 5 error): 28.974 9.67
  Train with distillation: [Epoch 79/100][Batch 0/5005]	 Loss 2.155, Top 1-error 32.422, Top 5-error 13.281
  Train with distillation: [Epoch 79/100][Batch 500/5005]	 Loss 2.073, Top 1-error 30.970, Top 5-error 12.499
  Train with distillation: [Epoch 79/100][Batch 1000/5005]	 Loss 2.077, Top 1-error 31.010, Top 5-error 12.551
  Train with distillation: [Epoch 79/100][Batch 1500/5005]	 Loss 2.078, Top 1-error 31.062, Top 5-error 12.552
  Train with distillation: [Epoch 79/100][Batch 2000/5005]	 Loss 2.077, Top 1-error 31.008, Top 5-error 12.507
  Train with distillation: [Epoch 79/100][Batch 2500/5005]	 Loss 2.079, Top 1-error 31.048, Top 5-error 12.525
  Train with distillation: [Epoch 79/100][Batch 3000/5005]	 Loss 2.079, Top 1-error 31.048, Top 5-error 12.528
  Train with distillation: [Epoch 79/100][Batch 3500/5005]	 Loss 2.080, Top 1-error 31.048, Top 5-error 12.540
  Train with distillation: [Epoch 79/100][Batch 4000/5005]	 Loss 2.081, Top 1-error 31.046, Top 5-error 12.550
  Train with distillation: [Epoch 79/100][Batch 4500/5005]	 Loss 2.081, Top 1-error 31.050, Top 5-error 12.545
  Train with distillation: [Epoch 79/100][Batch 5000/5005]	 Loss 2.080, Top 1-error 31.041, Top 5-error 12.520
  Train 	 Time Taken: 3164.57 sec
  Test (on val set): [Epoch 79/100][Batch 0/196]	Time 1.912 (1.912)	Loss 0.6567 (0.6567)	Top 1-err 18.3594 (18.3594)	Top 5-err 4.2969 (4.2969)
* Epoch: [79/100]	 Top 1-err 28.928  Top 5-err 9.722	 Test Loss 1.149
  Current best accuracy (top-1 and 5 error): 28.928 9.722
  Train with distillation: [Epoch 80/100][Batch 0/5005]	 Loss 2.211, Top 1-error 33.203, Top 5-error 14.062
  Train with distillation: [Epoch 80/100][Batch 500/5005]	 Loss 2.073, Top 1-error 30.874, Top 5-error 12.469
  Train with distillation: [Epoch 80/100][Batch 1000/5005]	 Loss 2.075, Top 1-error 30.954, Top 5-error 12.455
  Train with distillation: [Epoch 80/100][Batch 1500/5005]	 Loss 2.075, Top 1-error 30.918, Top 5-error 12.449
  Train with distillation: [Epoch 80/100][Batch 2000/5005]	 Loss 2.076, Top 1-error 30.961, Top 5-error 12.448
  Train with distillation: [Epoch 80/100][Batch 2500/5005]	 Loss 2.076, Top 1-error 30.993, Top 5-error 12.481
  Train with distillation: [Epoch 80/100][Batch 3000/5005]	 Loss 2.076, Top 1-error 31.012, Top 5-error 12.474
  Train with distillation: [Epoch 80/100][Batch 3500/5005]	 Loss 2.075, Top 1-error 30.998, Top 5-error 12.465
  Train with distillation: [Epoch 80/100][Batch 4000/5005]	 Loss 2.076, Top 1-error 31.000, Top 5-error 12.462
  Train with distillation: [Epoch 80/100][Batch 4500/5005]	 Loss 2.077, Top 1-error 31.015, Top 5-error 12.470
  Train with distillation: [Epoch 80/100][Batch 5000/5005]	 Loss 2.078, Top 1-error 31.032, Top 5-error 12.492
  Train 	 Time Taken: 3172.88 sec
  Test (on val set): [Epoch 80/100][Batch 0/196]	Time 2.054 (2.054)	Loss 0.6543 (0.6543)	Top 1-err 17.1875 (17.1875)	Top 5-err 3.1250 (3.1250)
* Epoch: [80/100]	 Top 1-err 28.942  Top 5-err 9.684	 Test Loss 1.147
  Current best accuracy (top-1 and 5 error): 28.928 9.722
  Train with distillation: [Epoch 81/100][Batch 0/5005]	 Loss 2.062, Top 1-error 30.859, Top 5-error 10.547
  Train with distillation: [Epoch 81/100][Batch 500/5005]	 Loss 2.068, Top 1-error 30.858, Top 5-error 12.356
  Train with distillation: [Epoch 81/100][Batch 1000/5005]	 Loss 2.077, Top 1-error 30.971, Top 5-error 12.459
  Train with distillation: [Epoch 81/100][Batch 1500/5005]	 Loss 2.075, Top 1-error 30.956, Top 5-error 12.450
  Train with distillation: [Epoch 81/100][Batch 2000/5005]	 Loss 2.075, Top 1-error 30.942, Top 5-error 12.457
  Train with distillation: [Epoch 81/100][Batch 2500/5005]	 Loss 2.074, Top 1-error 30.955, Top 5-error 12.448
  Train with distillation: [Epoch 81/100][Batch 3000/5005]	 Loss 2.075, Top 1-error 30.955, Top 5-error 12.446
  Train with distillation: [Epoch 81/100][Batch 3500/5005]	 Loss 2.075, Top 1-error 30.987, Top 5-error 12.472
  Train with distillation: [Epoch 81/100][Batch 4000/5005]	 Loss 2.076, Top 1-error 31.007, Top 5-error 12.475
  Train with distillation: [Epoch 81/100][Batch 4500/5005]	 Loss 2.075, Top 1-error 30.987, Top 5-error 12.457
  Train with distillation: [Epoch 81/100][Batch 5000/5005]	 Loss 2.076, Top 1-error 31.014, Top 5-error 12.482
  Train 	 Time Taken: 3172.22 sec
  Test (on val set): [Epoch 81/100][Batch 0/196]	Time 1.962 (1.962)	Loss 0.6242 (0.6242)	Top 1-err 16.7969 (16.7969)	Top 5-err 4.2969 (4.2969)
* Epoch: [81/100]	 Top 1-err 28.894  Top 5-err 9.722	 Test Loss 1.148
  Current best accuracy (top-1 and 5 error): 28.894 9.722
  Train with distillation: [Epoch 82/100][Batch 0/5005]	 Loss 2.116, Top 1-error 31.641, Top 5-error 11.719
  Train with distillation: [Epoch 82/100][Batch 500/5005]	 Loss 2.073, Top 1-error 31.121, Top 5-error 12.410
  Train with distillation: [Epoch 82/100][Batch 1000/5005]	 Loss 2.067, Top 1-error 30.934, Top 5-error 12.345
  Train with distillation: [Epoch 82/100][Batch 1500/5005]	 Loss 2.072, Top 1-error 30.998, Top 5-error 12.404
  Train with distillation: [Epoch 82/100][Batch 2000/5005]	 Loss 2.072, Top 1-error 31.030, Top 5-error 12.419
  Train with distillation: [Epoch 82/100][Batch 2500/5005]	 Loss 2.074, Top 1-error 31.024, Top 5-error 12.451
  Train with distillation: [Epoch 82/100][Batch 3000/5005]	 Loss 2.073, Top 1-error 31.006, Top 5-error 12.452
  Train with distillation: [Epoch 82/100][Batch 3500/5005]	 Loss 2.073, Top 1-error 30.998, Top 5-error 12.463
  Train with distillation: [Epoch 82/100][Batch 4000/5005]	 Loss 2.075, Top 1-error 31.036, Top 5-error 12.491
  Train with distillation: [Epoch 82/100][Batch 4500/5005]	 Loss 2.075, Top 1-error 31.022, Top 5-error 12.494
  Train with distillation: [Epoch 82/100][Batch 5000/5005]	 Loss 2.075, Top 1-error 31.021, Top 5-error 12.485
  Train 	 Time Taken: 3169.80 sec
  Test (on val set): [Epoch 82/100][Batch 0/196]	Time 2.062 (2.062)	Loss 0.6365 (0.6365)	Top 1-err 16.7969 (16.7969)	Top 5-err 4.2969 (4.2969)
* Epoch: [82/100]	 Top 1-err 28.872  Top 5-err 9.666	 Test Loss 1.150
  Current best accuracy (top-1 and 5 error): 28.872 9.666
  Train with distillation: [Epoch 83/100][Batch 0/5005]	 Loss 1.889, Top 1-error 28.906, Top 5-error 8.984
  Train with distillation: [Epoch 83/100][Batch 500/5005]	 Loss 2.063, Top 1-error 30.919, Top 5-error 12.422
  Train with distillation: [Epoch 83/100][Batch 1000/5005]	 Loss 2.064, Top 1-error 30.806, Top 5-error 12.393
  Train with distillation: [Epoch 83/100][Batch 1500/5005]	 Loss 2.066, Top 1-error 30.838, Top 5-error 12.394
  Train with distillation: [Epoch 83/100][Batch 2000/5005]	 Loss 2.068, Top 1-error 30.866, Top 5-error 12.385
  Train with distillation: [Epoch 83/100][Batch 2500/5005]	 Loss 2.069, Top 1-error 30.891, Top 5-error 12.396
  Train with distillation: [Epoch 83/100][Batch 3000/5005]	 Loss 2.070, Top 1-error 30.922, Top 5-error 12.400
  Train with distillation: [Epoch 83/100][Batch 3500/5005]	 Loss 2.069, Top 1-error 30.895, Top 5-error 12.411
  Train with distillation: [Epoch 83/100][Batch 4000/5005]	 Loss 2.072, Top 1-error 30.936, Top 5-error 12.449
  Train with distillation: [Epoch 83/100][Batch 4500/5005]	 Loss 2.072, Top 1-error 30.949, Top 5-error 12.441
  Train with distillation: [Epoch 83/100][Batch 5000/5005]	 Loss 2.072, Top 1-error 30.966, Top 5-error 12.451
  Train 	 Time Taken: 3165.63 sec
  Test (on val set): [Epoch 83/100][Batch 0/196]	Time 1.916 (1.916)	Loss 0.6115 (0.6115)	Top 1-err 16.0156 (16.0156)	Top 5-err 3.5156 (3.5156)
* Epoch: [83/100]	 Top 1-err 28.974  Top 5-err 9.714	 Test Loss 1.149
  Current best accuracy (top-1 and 5 error): 28.872 9.666
  Train with distillation: [Epoch 84/100][Batch 0/5005]	 Loss 2.186, Top 1-error 35.547, Top 5-error 15.625
  Train with distillation: [Epoch 84/100][Batch 500/5005]	 Loss 2.053, Top 1-error 30.622, Top 5-error 12.252
  Train with distillation: [Epoch 84/100][Batch 1000/5005]	 Loss 2.057, Top 1-error 30.591, Top 5-error 12.299
  Train with distillation: [Epoch 84/100][Batch 1500/5005]	 Loss 2.060, Top 1-error 30.687, Top 5-error 12.346
  Train with distillation: [Epoch 84/100][Batch 2000/5005]	 Loss 2.065, Top 1-error 30.758, Top 5-error 12.383
  Train with distillation: [Epoch 84/100][Batch 2500/5005]	 Loss 2.067, Top 1-error 30.798, Top 5-error 12.412
  Train with distillation: [Epoch 84/100][Batch 3000/5005]	 Loss 2.068, Top 1-error 30.837, Top 5-error 12.441
  Train with distillation: [Epoch 84/100][Batch 3500/5005]	 Loss 2.069, Top 1-error 30.852, Top 5-error 12.448
  Train with distillation: [Epoch 84/100][Batch 4000/5005]	 Loss 2.070, Top 1-error 30.860, Top 5-error 12.444
  Train with distillation: [Epoch 84/100][Batch 4500/5005]	 Loss 2.070, Top 1-error 30.872, Top 5-error 12.445
  Train with distillation: [Epoch 84/100][Batch 5000/5005]	 Loss 2.071, Top 1-error 30.890, Top 5-error 12.450
  Train 	 Time Taken: 3176.29 sec
  Test (on val set): [Epoch 84/100][Batch 0/196]	Time 2.054 (2.054)	Loss 0.6162 (0.6162)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.5156 (3.5156)
* Epoch: [84/100]	 Top 1-err 28.926  Top 5-err 9.616	 Test Loss 1.148
  Current best accuracy (top-1 and 5 error): 28.872 9.666
  Train with distillation: [Epoch 85/100][Batch 0/5005]	 Loss 1.943, Top 1-error 27.344, Top 5-error 9.375
  Train with distillation: [Epoch 85/100][Batch 500/5005]	 Loss 2.061, Top 1-error 30.746, Top 5-error 12.300
  Train with distillation: [Epoch 85/100][Batch 1000/5005]	 Loss 2.067, Top 1-error 30.889, Top 5-error 12.387
  Train with distillation: [Epoch 85/100][Batch 1500/5005]	 Loss 2.067, Top 1-error 30.863, Top 5-error 12.401
  Train with distillation: [Epoch 85/100][Batch 2000/5005]	 Loss 2.069, Top 1-error 30.907, Top 5-error 12.426
  Train with distillation: [Epoch 85/100][Batch 2500/5005]	 Loss 2.068, Top 1-error 30.868, Top 5-error 12.419
  Train with distillation: [Epoch 85/100][Batch 3000/5005]	 Loss 2.068, Top 1-error 30.868, Top 5-error 12.407
  Train with distillation: [Epoch 85/100][Batch 3500/5005]	 Loss 2.068, Top 1-error 30.872, Top 5-error 12.433
  Train with distillation: [Epoch 85/100][Batch 4000/5005]	 Loss 2.069, Top 1-error 30.872, Top 5-error 12.435
  Train with distillation: [Epoch 85/100][Batch 4500/5005]	 Loss 2.070, Top 1-error 30.871, Top 5-error 12.438
  Train with distillation: [Epoch 85/100][Batch 5000/5005]	 Loss 2.070, Top 1-error 30.890, Top 5-error 12.455
  Train 	 Time Taken: 3179.69 sec
  Test (on val set): [Epoch 85/100][Batch 0/196]	Time 1.893 (1.893)	Loss 0.5908 (0.5908)	Top 1-err 17.1875 (17.1875)	Top 5-err 2.3438 (2.3438)
* Epoch: [85/100]	 Top 1-err 28.962  Top 5-err 9.680	 Test Loss 1.147
  Current best accuracy (top-1 and 5 error): 28.872 9.666
  Train with distillation: [Epoch 86/100][Batch 0/5005]	 Loss 1.909, Top 1-error 29.297, Top 5-error 12.109
  Train with distillation: [Epoch 86/100][Batch 500/5005]	 Loss 2.057, Top 1-error 30.596, Top 5-error 12.187
  Train with distillation: [Epoch 86/100][Batch 1000/5005]	 Loss 2.063, Top 1-error 30.753, Top 5-error 12.367
  Train with distillation: [Epoch 86/100][Batch 1500/5005]	 Loss 2.065, Top 1-error 30.793, Top 5-error 12.378
  Train with distillation: [Epoch 86/100][Batch 2000/5005]	 Loss 2.065, Top 1-error 30.760, Top 5-error 12.398
  Train with distillation: [Epoch 86/100][Batch 2500/5005]	 Loss 2.068, Top 1-error 30.835, Top 5-error 12.410
  Train with distillation: [Epoch 86/100][Batch 3000/5005]	 Loss 2.068, Top 1-error 30.860, Top 5-error 12.419
  Train with distillation: [Epoch 86/100][Batch 3500/5005]	 Loss 2.068, Top 1-error 30.873, Top 5-error 12.405
  Train with distillation: [Epoch 86/100][Batch 4000/5005]	 Loss 2.069, Top 1-error 30.896, Top 5-error 12.419
  Train with distillation: [Epoch 86/100][Batch 4500/5005]	 Loss 2.068, Top 1-error 30.895, Top 5-error 12.402
  Train with distillation: [Epoch 86/100][Batch 5000/5005]	 Loss 2.069, Top 1-error 30.914, Top 5-error 12.408
  Train 	 Time Taken: 3174.60 sec
  Test (on val set): [Epoch 86/100][Batch 0/196]	Time 2.079 (2.079)	Loss 0.5943 (0.5943)	Top 1-err 16.4062 (16.4062)	Top 5-err 3.1250 (3.1250)
* Epoch: [86/100]	 Top 1-err 29.038  Top 5-err 9.692	 Test Loss 1.148
  Current best accuracy (top-1 and 5 error): 28.872 9.666
  Train with distillation: [Epoch 87/100][Batch 0/5005]	 Loss 2.054, Top 1-error 30.078, Top 5-error 11.328
  Train with distillation: [Epoch 87/100][Batch 500/5005]	 Loss 2.067, Top 1-error 30.838, Top 5-error 12.313
  Train with distillation: [Epoch 87/100][Batch 1000/5005]	 Loss 2.064, Top 1-error 30.670, Top 5-error 12.338
  Train with distillation: [Epoch 87/100][Batch 1500/5005]	 Loss 2.065, Top 1-error 30.749, Top 5-error 12.397
  Train with distillation: [Epoch 87/100][Batch 2000/5005]	 Loss 2.065, Top 1-error 30.764, Top 5-error 12.366
  Train with distillation: [Epoch 87/100][Batch 2500/5005]	 Loss 2.066, Top 1-error 30.776, Top 5-error 12.381
  Train with distillation: [Epoch 87/100][Batch 3000/5005]	 Loss 2.069, Top 1-error 30.821, Top 5-error 12.412
  Train with distillation: [Epoch 87/100][Batch 3500/5005]	 Loss 2.068, Top 1-error 30.811, Top 5-error 12.413
  Train with distillation: [Epoch 87/100][Batch 4000/5005]	 Loss 2.068, Top 1-error 30.843, Top 5-error 12.418
  Train with distillation: [Epoch 87/100][Batch 4500/5005]	 Loss 2.068, Top 1-error 30.847, Top 5-error 12.430
  Train with distillation: [Epoch 87/100][Batch 5000/5005]	 Loss 2.068, Top 1-error 30.831, Top 5-error 12.427
  Train 	 Time Taken: 3167.84 sec
  Test (on val set): [Epoch 87/100][Batch 0/196]	Time 1.946 (1.946)	Loss 0.5919 (0.5919)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.5156 (3.5156)
* Epoch: [87/100]	 Top 1-err 28.852  Top 5-err 9.654	 Test Loss 1.146
  Current best accuracy (top-1 and 5 error): 28.852 9.654
  Train with distillation: [Epoch 88/100][Batch 0/5005]	 Loss 1.938, Top 1-error 28.125, Top 5-error 9.375
  Train with distillation: [Epoch 88/100][Batch 500/5005]	 Loss 2.060, Top 1-error 30.707, Top 5-error 12.411
  Train with distillation: [Epoch 88/100][Batch 1000/5005]	 Loss 2.062, Top 1-error 30.702, Top 5-error 12.375
  Train with distillation: [Epoch 88/100][Batch 1500/5005]	 Loss 2.063, Top 1-error 30.748, Top 5-error 12.351
  Train with distillation: [Epoch 88/100][Batch 2000/5005]	 Loss 2.063, Top 1-error 30.781, Top 5-error 12.341
  Train with distillation: [Epoch 88/100][Batch 2500/5005]	 Loss 2.063, Top 1-error 30.837, Top 5-error 12.339
  Train with distillation: [Epoch 88/100][Batch 3000/5005]	 Loss 2.063, Top 1-error 30.806, Top 5-error 12.330
  Train with distillation: [Epoch 88/100][Batch 3500/5005]	 Loss 2.065, Top 1-error 30.827, Top 5-error 12.333
  Train with distillation: [Epoch 88/100][Batch 4000/5005]	 Loss 2.064, Top 1-error 30.808, Top 5-error 12.324
  Train with distillation: [Epoch 88/100][Batch 4500/5005]	 Loss 2.066, Top 1-error 30.845, Top 5-error 12.346
  Train with distillation: [Epoch 88/100][Batch 5000/5005]	 Loss 2.065, Top 1-error 30.819, Top 5-error 12.345
  Train 	 Time Taken: 3164.41 sec
  Test (on val set): [Epoch 88/100][Batch 0/196]	Time 2.037 (2.037)	Loss 0.6168 (0.6168)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.9062 (3.9062)
* Epoch: [88/100]	 Top 1-err 28.864  Top 5-err 9.702	 Test Loss 1.149
  Current best accuracy (top-1 and 5 error): 28.852 9.654
  Train with distillation: [Epoch 89/100][Batch 0/5005]	 Loss 1.899, Top 1-error 27.734, Top 5-error 9.766
  Train with distillation: [Epoch 89/100][Batch 500/5005]	 Loss 2.057, Top 1-error 30.662, Top 5-error 12.297
  Train with distillation: [Epoch 89/100][Batch 1000/5005]	 Loss 2.063, Top 1-error 30.786, Top 5-error 12.411
  Train with distillation: [Epoch 89/100][Batch 1500/5005]	 Loss 2.064, Top 1-error 30.787, Top 5-error 12.438
  Train with distillation: [Epoch 89/100][Batch 2000/5005]	 Loss 2.064, Top 1-error 30.747, Top 5-error 12.402
  Train with distillation: [Epoch 89/100][Batch 2500/5005]	 Loss 2.064, Top 1-error 30.764, Top 5-error 12.394
  Train with distillation: [Epoch 89/100][Batch 3000/5005]	 Loss 2.063, Top 1-error 30.738, Top 5-error 12.364
  Train with distillation: [Epoch 89/100][Batch 3500/5005]	 Loss 2.064, Top 1-error 30.772, Top 5-error 12.392
  Train with distillation: [Epoch 89/100][Batch 4000/5005]	 Loss 2.065, Top 1-error 30.790, Top 5-error 12.413
  Train with distillation: [Epoch 89/100][Batch 4500/5005]	 Loss 2.065, Top 1-error 30.788, Top 5-error 12.403
  Train with distillation: [Epoch 89/100][Batch 5000/5005]	 Loss 2.066, Top 1-error 30.794, Top 5-error 12.410
  Train 	 Time Taken: 3160.45 sec
  Test (on val set): [Epoch 89/100][Batch 0/196]	Time 1.948 (1.948)	Loss 0.6224 (0.6224)	Top 1-err 17.9688 (17.9688)	Top 5-err 3.5156 (3.5156)
* Epoch: [89/100]	 Top 1-err 28.858  Top 5-err 9.668	 Test Loss 1.144
  Current best accuracy (top-1 and 5 error): 28.852 9.654
  Train with distillation: [Epoch 90/100][Batch 0/5005]	 Loss 2.075, Top 1-error 29.297, Top 5-error 12.500
  Train with distillation: [Epoch 90/100][Batch 500/5005]	 Loss 2.048, Top 1-error 30.544, Top 5-error 12.308
  Train with distillation: [Epoch 90/100][Batch 1000/5005]	 Loss 2.041, Top 1-error 30.434, Top 5-error 12.235
  Train with distillation: [Epoch 90/100][Batch 1500/5005]	 Loss 2.038, Top 1-error 30.429, Top 5-error 12.184
  Train with distillation: [Epoch 90/100][Batch 2000/5005]	 Loss 2.033, Top 1-error 30.410, Top 5-error 12.130
  Train with distillation: [Epoch 90/100][Batch 2500/5005]	 Loss 2.032, Top 1-error 30.370, Top 5-error 12.139
  Train with distillation: [Epoch 90/100][Batch 3000/5005]	 Loss 2.031, Top 1-error 30.348, Top 5-error 12.138
  Train with distillation: [Epoch 90/100][Batch 3500/5005]	 Loss 2.031, Top 1-error 30.349, Top 5-error 12.131
  Train with distillation: [Epoch 90/100][Batch 4000/5005]	 Loss 2.031, Top 1-error 30.379, Top 5-error 12.130
  Train with distillation: [Epoch 90/100][Batch 4500/5005]	 Loss 2.031, Top 1-error 30.369, Top 5-error 12.132
  Train with distillation: [Epoch 90/100][Batch 5000/5005]	 Loss 2.030, Top 1-error 30.351, Top 5-error 12.125
  Train 	 Time Taken: 3147.80 sec
  Test (on val set): [Epoch 90/100][Batch 0/196]	Time 1.979 (1.979)	Loss 0.5900 (0.5900)	Top 1-err 17.1875 (17.1875)	Top 5-err 3.5156 (3.5156)
* Epoch: [90/100]	 Top 1-err 28.548  Top 5-err 9.598	 Test Loss 1.131
  Current best accuracy (top-1 and 5 error): 28.548 9.598
  Train with distillation: [Epoch 91/100][Batch 0/5005]	 Loss 1.946, Top 1-error 27.734, Top 5-error 12.500
  Train with distillation: [Epoch 91/100][Batch 500/5005]	 Loss 2.030, Top 1-error 30.414, Top 5-error 12.157
  Train with distillation: [Epoch 91/100][Batch 1000/5005]	 Loss 2.029, Top 1-error 30.416, Top 5-error 12.130
  Train with distillation: [Epoch 91/100][Batch 1500/5005]	 Loss 2.029, Top 1-error 30.407, Top 5-error 12.140
  Train with distillation: [Epoch 91/100][Batch 2000/5005]	 Loss 2.027, Top 1-error 30.368, Top 5-error 12.105
  Train with distillation: [Epoch 91/100][Batch 2500/5005]	 Loss 2.026, Top 1-error 30.385, Top 5-error 12.101
  Train with distillation: [Epoch 91/100][Batch 3000/5005]	 Loss 2.026, Top 1-error 30.399, Top 5-error 12.107
  Train with distillation: [Epoch 91/100][Batch 3500/5005]	 Loss 2.026, Top 1-error 30.400, Top 5-error 12.104
  Train with distillation: [Epoch 91/100][Batch 4000/5005]	 Loss 2.026, Top 1-error 30.399, Top 5-error 12.115
  Train with distillation: [Epoch 91/100][Batch 4500/5005]	 Loss 2.027, Top 1-error 30.396, Top 5-error 12.118
  Train with distillation: [Epoch 91/100][Batch 5000/5005]	 Loss 2.026, Top 1-error 30.381, Top 5-error 12.120
  Train 	 Time Taken: 3145.77 sec
  Test (on val set): [Epoch 91/100][Batch 0/196]	Time 1.905 (1.905)	Loss 0.6113 (0.6113)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.5156 (3.5156)
* Epoch: [91/100]	 Top 1-err 28.502  Top 5-err 9.534	 Test Loss 1.129
  Current best accuracy (top-1 and 5 error): 28.502 9.534
  Train with distillation: [Epoch 92/100][Batch 0/5005]	 Loss 2.048, Top 1-error 30.859, Top 5-error 12.891
  Train with distillation: [Epoch 92/100][Batch 500/5005]	 Loss 2.016, Top 1-error 30.214, Top 5-error 12.060
  Train with distillation: [Epoch 92/100][Batch 1000/5005]	 Loss 2.017, Top 1-error 30.206, Top 5-error 12.013
  Train with distillation: [Epoch 92/100][Batch 1500/5005]	 Loss 2.019, Top 1-error 30.250, Top 5-error 12.020
  Train with distillation: [Epoch 92/100][Batch 2000/5005]	 Loss 2.019, Top 1-error 30.246, Top 5-error 12.025
  Train with distillation: [Epoch 92/100][Batch 2500/5005]	 Loss 2.020, Top 1-error 30.239, Top 5-error 12.050
  Train with distillation: [Epoch 92/100][Batch 3000/5005]	 Loss 2.021, Top 1-error 30.259, Top 5-error 12.059
  Train with distillation: [Epoch 92/100][Batch 3500/5005]	 Loss 2.020, Top 1-error 30.245, Top 5-error 12.055
  Train with distillation: [Epoch 92/100][Batch 4000/5005]	 Loss 2.020, Top 1-error 30.241, Top 5-error 12.039
  Train with distillation: [Epoch 92/100][Batch 4500/5005]	 Loss 2.020, Top 1-error 30.238, Top 5-error 12.040
  Train with distillation: [Epoch 92/100][Batch 5000/5005]	 Loss 2.020, Top 1-error 30.234, Top 5-error 12.043
  Train 	 Time Taken: 3153.98 sec
  Test (on val set): [Epoch 92/100][Batch 0/196]	Time 2.012 (2.012)	Loss 0.6044 (0.6044)	Top 1-err 16.7969 (16.7969)	Top 5-err 3.5156 (3.5156)
* Epoch: [92/100]	 Top 1-err 28.512  Top 5-err 9.618	 Test Loss 1.129
  Current best accuracy (top-1 and 5 error): 28.502 9.534
  Train with distillation: [Epoch 93/100][Batch 0/5005]	 Loss 2.090, Top 1-error 30.469, Top 5-error 14.453
  Train with distillation: [Epoch 93/100][Batch 500/5005]	 Loss 2.019, Top 1-error 30.172, Top 5-error 12.042
  Train with distillation: [Epoch 93/100][Batch 1000/5005]	 Loss 2.019, Top 1-error 30.189, Top 5-error 12.062
  Train with distillation: [Epoch 93/100][Batch 1500/5005]	 Loss 2.018, Top 1-error 30.156, Top 5-error 12.072
  Train with distillation: [Epoch 93/100][Batch 2000/5005]	 Loss 2.021, Top 1-error 30.216, Top 5-error 12.102
  Train with distillation: [Epoch 93/100][Batch 2500/5005]	 Loss 2.020, Top 1-error 30.228, Top 5-error 12.083
  Train with distillation: [Epoch 93/100][Batch 3000/5005]	 Loss 2.020, Top 1-error 30.247, Top 5-error 12.088
  Train with distillation: [Epoch 93/100][Batch 3500/5005]	 Loss 2.019, Top 1-error 30.235, Top 5-error 12.071
  Train with distillation: [Epoch 93/100][Batch 4000/5005]	 Loss 2.018, Top 1-error 30.217, Top 5-error 12.050
  Train with distillation: [Epoch 93/100][Batch 4500/5005]	 Loss 2.020, Top 1-error 30.256, Top 5-error 12.069
  Train with distillation: [Epoch 93/100][Batch 5000/5005]	 Loss 2.020, Top 1-error 30.253, Top 5-error 12.075
  Train 	 Time Taken: 3159.54 sec
  Test (on val set): [Epoch 93/100][Batch 0/196]	Time 1.912 (1.912)	Loss 0.6097 (0.6097)	Top 1-err 15.6250 (15.6250)	Top 5-err 3.9062 (3.9062)
* Epoch: [93/100]	 Top 1-err 28.448  Top 5-err 9.554	 Test Loss 1.130
  Current best accuracy (top-1 and 5 error): 28.448 9.554
  Train with distillation: [Epoch 94/100][Batch 0/5005]	 Loss 1.977, Top 1-error 29.688, Top 5-error 14.062
  Train with distillation: [Epoch 94/100][Batch 500/5005]	 Loss 2.017, Top 1-error 30.165, Top 5-error 12.115
  Train with distillation: [Epoch 94/100][Batch 1000/5005]	 Loss 2.021, Top 1-error 30.261, Top 5-error 12.170
  Train with distillation: [Epoch 94/100][Batch 1500/5005]	 Loss 2.019, Top 1-error 30.263, Top 5-error 12.101
  Train with distillation: [Epoch 94/100][Batch 2000/5005]	 Loss 2.019, Top 1-error 30.223, Top 5-error 12.120
  Train with distillation: [Epoch 94/100][Batch 2500/5005]	 Loss 2.020, Top 1-error 30.275, Top 5-error 12.134
  Train with distillation: [Epoch 94/100][Batch 3000/5005]	 Loss 2.020, Top 1-error 30.243, Top 5-error 12.115
  Train with distillation: [Epoch 94/100][Batch 3500/5005]	 Loss 2.019, Top 1-error 30.236, Top 5-error 12.101
  Train with distillation: [Epoch 94/100][Batch 4000/5005]	 Loss 2.019, Top 1-error 30.206, Top 5-error 12.097
  Train with distillation: [Epoch 94/100][Batch 4500/5005]	 Loss 2.019, Top 1-error 30.221, Top 5-error 12.092
  Train with distillation: [Epoch 94/100][Batch 5000/5005]	 Loss 2.020, Top 1-error 30.222, Top 5-error 12.099
  Train 	 Time Taken: 3168.99 sec
  Test (on val set): [Epoch 94/100][Batch 0/196]	Time 1.987 (1.987)	Loss 0.5968 (0.5968)	Top 1-err 16.0156 (16.0156)	Top 5-err 4.2969 (4.2969)
* Epoch: [94/100]	 Top 1-err 28.532  Top 5-err 9.510	 Test Loss 1.129
  Current best accuracy (top-1 and 5 error): 28.448 9.554
  Train with distillation: [Epoch 95/100][Batch 0/5005]	 Loss 1.888, Top 1-error 28.516, Top 5-error 10.156
  Train with distillation: [Epoch 95/100][Batch 500/5005]	 Loss 2.017, Top 1-error 30.071, Top 5-error 12.134
  Train with distillation: [Epoch 95/100][Batch 1000/5005]	 Loss 2.016, Top 1-error 30.133, Top 5-error 12.115
  Train with distillation: [Epoch 95/100][Batch 1500/5005]	 Loss 2.017, Top 1-error 30.247, Top 5-error 12.121
  Train with distillation: [Epoch 95/100][Batch 2000/5005]	 Loss 2.015, Top 1-error 30.182, Top 5-error 12.060
  Train with distillation: [Epoch 95/100][Batch 2500/5005]	 Loss 2.016, Top 1-error 30.192, Top 5-error 12.062
  Train with distillation: [Epoch 95/100][Batch 3000/5005]	 Loss 2.017, Top 1-error 30.206, Top 5-error 12.085
  Train with distillation: [Epoch 95/100][Batch 3500/5005]	 Loss 2.017, Top 1-error 30.216, Top 5-error 12.071
  Train with distillation: [Epoch 95/100][Batch 4000/5005]	 Loss 2.017, Top 1-error 30.206, Top 5-error 12.058
  Train with distillation: [Epoch 95/100][Batch 4500/5005]	 Loss 2.017, Top 1-error 30.214, Top 5-error 12.070
  Train with distillation: [Epoch 95/100][Batch 5000/5005]	 Loss 2.017, Top 1-error 30.220, Top 5-error 12.073
  Train 	 Time Taken: 3180.51 sec
  Test (on val set): [Epoch 95/100][Batch 0/196]	Time 1.918 (1.918)	Loss 0.6085 (0.6085)	Top 1-err 16.4062 (16.4062)	Top 5-err 3.9062 (3.9062)
* Epoch: [95/100]	 Top 1-err 28.514  Top 5-err 9.516	 Test Loss 1.129
  Current best accuracy (top-1 and 5 error): 28.448 9.554
  Train with distillation: [Epoch 96/100][Batch 0/5005]	 Loss 1.829, Top 1-error 27.734, Top 5-error 8.984
  Train with distillation: [Epoch 96/100][Batch 500/5005]	 Loss 2.024, Top 1-error 30.344, Top 5-error 12.105
  Train with distillation: [Epoch 96/100][Batch 1000/5005]	 Loss 2.018, Top 1-error 30.240, Top 5-error 12.079
  Train with distillation: [Epoch 96/100][Batch 1500/5005]	 Loss 2.023, Top 1-error 30.302, Top 5-error 12.150
  Train with distillation: [Epoch 96/100][Batch 2000/5005]	 Loss 2.020, Top 1-error 30.264, Top 5-error 12.100
  Train with distillation: [Epoch 96/100][Batch 2500/5005]	 Loss 2.019, Top 1-error 30.211, Top 5-error 12.080
  Train with distillation: [Epoch 96/100][Batch 3000/5005]	 Loss 2.018, Top 1-error 30.185, Top 5-error 12.080
  Train with distillation: [Epoch 96/100][Batch 3500/5005]	 Loss 2.018, Top 1-error 30.196, Top 5-error 12.089
  Train with distillation: [Epoch 96/100][Batch 4000/5005]	 Loss 2.018, Top 1-error 30.212, Top 5-error 12.085
  Train with distillation: [Epoch 96/100][Batch 4500/5005]	 Loss 2.018, Top 1-error 30.213, Top 5-error 12.086
  Train with distillation: [Epoch 96/100][Batch 5000/5005]	 Loss 2.018, Top 1-error 30.209, Top 5-error 12.079
  Train 	 Time Taken: 3190.93 sec
  Test (on val set): [Epoch 96/100][Batch 0/196]	Time 2.054 (2.054)	Loss 0.6019 (0.6019)	Top 1-err 16.4062 (16.4062)	Top 5-err 3.9062 (3.9062)
* Epoch: [96/100]	 Top 1-err 28.500  Top 5-err 9.538	 Test Loss 1.129
  Current best accuracy (top-1 and 5 error): 28.448 9.554
  Train with distillation: [Epoch 97/100][Batch 0/5005]	 Loss 2.219, Top 1-error 34.375, Top 5-error 13.281
  Train with distillation: [Epoch 97/100][Batch 500/5005]	 Loss 2.006, Top 1-error 29.929, Top 5-error 12.040
  Train with distillation: [Epoch 97/100][Batch 1000/5005]	 Loss 2.012, Top 1-error 30.047, Top 5-error 12.020
  Train with distillation: [Epoch 97/100][Batch 1500/5005]	 Loss 2.011, Top 1-error 30.038, Top 5-error 12.007
  Train with distillation: [Epoch 97/100][Batch 2000/5005]	 Loss 2.013, Top 1-error 30.086, Top 5-error 12.018
  Train with distillation: [Epoch 97/100][Batch 2500/5005]	 Loss 2.013, Top 1-error 30.110, Top 5-error 12.008
  Train with distillation: [Epoch 97/100][Batch 3000/5005]	 Loss 2.014, Top 1-error 30.144, Top 5-error 12.018
  Train with distillation: [Epoch 97/100][Batch 3500/5005]	 Loss 2.015, Top 1-error 30.175, Top 5-error 12.026
  Train with distillation: [Epoch 97/100][Batch 4000/5005]	 Loss 2.015, Top 1-error 30.169, Top 5-error 12.024
  Train with distillation: [Epoch 97/100][Batch 4500/5005]	 Loss 2.016, Top 1-error 30.173, Top 5-error 12.026
  Train with distillation: [Epoch 97/100][Batch 5000/5005]	 Loss 2.016, Top 1-error 30.174, Top 5-error 12.021
  Train 	 Time Taken: 3185.55 sec
  Test (on val set): [Epoch 97/100][Batch 0/196]	Time 1.912 (1.912)	Loss 0.6026 (0.6026)	Top 1-err 16.4062 (16.4062)	Top 5-err 4.2969 (4.2969)
* Epoch: [97/100]	 Top 1-err 28.456  Top 5-err 9.510	 Test Loss 1.128
  Current best accuracy (top-1 and 5 error): 28.448 9.554
  Train with distillation: [Epoch 98/100][Batch 0/5005]	 Loss 1.937, Top 1-error 26.953, Top 5-error 12.891
  Train with distillation: [Epoch 98/100][Batch 500/5005]	 Loss 2.015, Top 1-error 30.141, Top 5-error 12.025
  Train with distillation: [Epoch 98/100][Batch 1000/5005]	 Loss 2.017, Top 1-error 30.173, Top 5-error 12.063
  Train with distillation: [Epoch 98/100][Batch 1500/5005]	 Loss 2.017, Top 1-error 30.183, Top 5-error 12.016
  Train with distillation: [Epoch 98/100][Batch 2000/5005]	 Loss 2.015, Top 1-error 30.120, Top 5-error 12.000
  Train with distillation: [Epoch 98/100][Batch 2500/5005]	 Loss 2.016, Top 1-error 30.146, Top 5-error 12.035
  Train with distillation: [Epoch 98/100][Batch 3000/5005]	 Loss 2.017, Top 1-error 30.178, Top 5-error 12.066
  Train with distillation: [Epoch 98/100][Batch 3500/5005]	 Loss 2.016, Top 1-error 30.173, Top 5-error 12.056
  Train with distillation: [Epoch 98/100][Batch 4000/5005]	 Loss 2.016, Top 1-error 30.164, Top 5-error 12.048
  Train with distillation: [Epoch 98/100][Batch 4500/5005]	 Loss 2.015, Top 1-error 30.162, Top 5-error 12.037
  Train with distillation: [Epoch 98/100][Batch 5000/5005]	 Loss 2.016, Top 1-error 30.168, Top 5-error 12.051
  Train 	 Time Taken: 3181.31 sec
  Test (on val set): [Epoch 98/100][Batch 0/196]	Time 1.980 (1.980)	Loss 0.6082 (0.6082)	Top 1-err 16.4062 (16.4062)	Top 5-err 4.2969 (4.2969)
* Epoch: [98/100]	 Top 1-err 28.428  Top 5-err 9.496	 Test Loss 1.128
  Current best accuracy (top-1 and 5 error): 28.428 9.496
  Train with distillation: [Epoch 99/100][Batch 0/5005]	 Loss 2.133, Top 1-error 30.078, Top 5-error 13.281
  Train with distillation: [Epoch 99/100][Batch 500/5005]	 Loss 2.020, Top 1-error 30.148, Top 5-error 12.036
  Train with distillation: [Epoch 99/100][Batch 1000/5005]	 Loss 2.018, Top 1-error 30.188, Top 5-error 12.074
  Train with distillation: [Epoch 99/100][Batch 1500/5005]	 Loss 2.016, Top 1-error 30.143, Top 5-error 12.058
  Train with distillation: [Epoch 99/100][Batch 2000/5005]	 Loss 2.015, Top 1-error 30.103, Top 5-error 12.061
  Train with distillation: [Epoch 99/100][Batch 2500/5005]	 Loss 2.013, Top 1-error 30.084, Top 5-error 12.041
  Train with distillation: [Epoch 99/100][Batch 3000/5005]	 Loss 2.013, Top 1-error 30.088, Top 5-error 12.044
  Train with distillation: [Epoch 99/100][Batch 3500/5005]	 Loss 2.013, Top 1-error 30.106, Top 5-error 12.038
  Train with distillation: [Epoch 99/100][Batch 4000/5005]	 Loss 2.014, Top 1-error 30.118, Top 5-error 12.045
  Train with distillation: [Epoch 99/100][Batch 4500/5005]	 Loss 2.014, Top 1-error 30.134, Top 5-error 12.051
  Train with distillation: [Epoch 99/100][Batch 5000/5005]	 Loss 2.015, Top 1-error 30.168, Top 5-error 12.067
  Train 	 Time Taken: 3211.22 sec
  Test (on val set): [Epoch 99/100][Batch 0/196]	Time 1.955 (1.955)	Loss 0.6016 (0.6016)	Top 1-err 15.6250 (15.6250)	Top 5-err 4.2969 (4.2969)
* Epoch: [99/100]	 Top 1-err 28.438  Top 5-err 9.502	 Test Loss 1.128
  Current best accuracy (top-1 and 5 error): 28.428 9.496
  Train with distillation: [Epoch 100/100][Batch 0/5005]	 Loss 1.870, Top 1-error 25.781, Top 5-error 10.547
  Train with distillation: [Epoch 100/100][Batch 500/5005]	 Loss 2.019, Top 1-error 30.516, Top 5-error 12.081
  Train with distillation: [Epoch 100/100][Batch 1000/5005]	 Loss 2.016, Top 1-error 30.274, Top 5-error 12.063
  Train with distillation: [Epoch 100/100][Batch 1500/5005]	 Loss 2.015, Top 1-error 30.251, Top 5-error 12.044
  Train with distillation: [Epoch 100/100][Batch 2000/5005]	 Loss 2.013, Top 1-error 30.229, Top 5-error 12.012
  Train with distillation: [Epoch 100/100][Batch 2500/5005]	 Loss 2.013, Top 1-error 30.184, Top 5-error 12.003
  Train with distillation: [Epoch 100/100][Batch 3000/5005]	 Loss 2.013, Top 1-error 30.176, Top 5-error 12.000
  Train with distillation: [Epoch 100/100][Batch 3500/5005]	 Loss 2.014, Top 1-error 30.185, Top 5-error 12.028
  Train with distillation: [Epoch 100/100][Batch 4000/5005]	 Loss 2.015, Top 1-error 30.195, Top 5-error 12.034
  Train with distillation: [Epoch 100/100][Batch 4500/5005]	 Loss 2.015, Top 1-error 30.202, Top 5-error 12.035
  Train with distillation: [Epoch 100/100][Batch 5000/5005]	 Loss 2.015, Top 1-error 30.187, Top 5-error 12.046
  Train 	 Time Taken: 3200.31 sec
  Test (on val set): [Epoch 100/100][Batch 0/196]	Time 2.074 (2.074)	Loss 0.5955 (0.5955)	Top 1-err 16.0156 (16.0156)	Top 5-err 3.9062 (3.9062)
* Epoch: [100/100]	 Top 1-err 28.430  Top 5-err 9.522	 Test Loss 1.126
  Current best accuracy (top-1 and 5 error): 28.428 9.496
  Best accuracy (top-1 and 5 error): 28.428 9.496