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
Test (on val set): [Epoch 0/100][Batch 0/196]	Time 3.074 (3.074)	Loss 0.5336 (0.5336)	Top 1-err 16.7969 (16.7969)	Top 5-err 2.3438 (2.3438)
* Epoch: [0/100]	 Top 1-err 26.688  Top 5-err 8.580	 Test Loss 1.081
Train with distillation: [Epoch 1/100][Batch 0/5005]	 Loss 15.053, Top 1-error 100.000, Top 5-error 100.000
Train with distillation: [Epoch 1/100][Batch 500/5005]	 Loss 12.376, Top 1-error 97.149, Top 5-error 90.458
Train with distillation: [Epoch 1/100][Batch 1000/5005]	 Loss 11.361, Top 1-error 94.477, Top 5-error 83.904
Train with distillation: [Epoch 1/100][Batch 1500/5005]	 Loss 10.673, Top 1-error 91.926, Top 5-error 78.568
Train with distillation: [Epoch 1/100][Batch 2000/5005]	 Loss 10.140, Top 1-error 89.544, Top 5-error 74.147
Train with distillation: [Epoch 1/100][Batch 2500/5005]	 Loss 9.703, Top 1-error 87.453, Top 5-error 70.457
Train with distillation: [Epoch 1/100][Batch 3000/5005]	 Loss 9.337, Top 1-error 85.535, Top 5-error 67.333
Train with distillation: [Epoch 1/100][Batch 3500/5005]	 Loss 9.023, Top 1-error 83.828, Top 5-error 64.683
Train with distillation: [Epoch 1/100][Batch 4000/5005]	 Loss 8.751, Top 1-error 82.273, Top 5-error 62.389
Train with distillation: [Epoch 1/100][Batch 4500/5005]	 Loss 8.512, Top 1-error 80.882, Top 5-error 60.373
Train with distillation: [Epoch 1/100][Batch 5000/5005]	 Loss 8.297, Top 1-error 79.599, Top 5-error 58.558
Train 	 Time Taken: 3117.36 sec
Test (on val set): [Epoch 1/100][Batch 0/196]	Time 1.815 (1.815)	Loss 1.9821 (1.9821)	Top 1-err 51.9531 (51.9531)	Top 5-err 20.3125 (20.3125)
* Epoch: [1/100]	 Top 1-err 65.834  Top 5-err 38.298	 Test Loss 2.988
Current best accuracy (top-1 and 5 error): 65.834 38.298
Train with distillation: [Epoch 2/100][Batch 0/5005]	 Loss 6.307, Top 1-error 65.234, Top 5-error 38.281
Train with distillation: [Epoch 2/100][Batch 500/5005]	 Loss 6.167, Top 1-error 66.401, Top 5-error 40.548
Train with distillation: [Epoch 2/100][Batch 1000/5005]	 Loss 6.087, Top 1-error 65.913, Top 5-error 40.053
Train with distillation: [Epoch 2/100][Batch 1500/5005]	 Loss 6.013, Top 1-error 65.364, Top 5-error 39.453
Train with distillation: [Epoch 2/100][Batch 2000/5005]	 Loss 5.945, Top 1-error 64.873, Top 5-error 38.939
Train with distillation: [Epoch 2/100][Batch 2500/5005]	 Loss 5.886, Top 1-error 64.492, Top 5-error 38.551
Train with distillation: [Epoch 2/100][Batch 3000/5005]	 Loss 5.828, Top 1-error 64.040, Top 5-error 38.124
Train with distillation: [Epoch 2/100][Batch 3500/5005]	 Loss 5.773, Top 1-error 63.645, Top 5-error 37.723
Train with distillation: [Epoch 2/100][Batch 4000/5005]	 Loss 5.723, Top 1-error 63.276, Top 5-error 37.368
Train with distillation: [Epoch 2/100][Batch 4500/5005]	 Loss 5.675, Top 1-error 62.941, Top 5-error 37.025
Train with distillation: [Epoch 2/100][Batch 5000/5005]	 Loss 5.630, Top 1-error 62.602, Top 5-error 36.704
Train 	 Time Taken: 3185.57 sec
Test (on val set): [Epoch 2/100][Batch 0/196]	Time 1.988 (1.988)	Loss 1.7629 (1.7629)	Top 1-err 41.4062 (41.4062)	Top 5-err 15.6250 (15.6250)
* Epoch: [2/100]	 Top 1-err 57.172  Top 5-err 29.286	 Test Loss 2.494
Current best accuracy (top-1 and 5 error): 57.172 29.286
Train with distillation: [Epoch 3/100][Batch 0/5005]	 Loss 5.003, Top 1-error 58.203, Top 5-error 30.078
Train with distillation: [Epoch 3/100][Batch 500/5005]	 Loss 5.096, Top 1-error 58.641, Top 5-error 32.705
Train with distillation: [Epoch 3/100][Batch 1000/5005]	 Loss 5.077, Top 1-error 58.481, Top 5-error 32.478
Train with distillation: [Epoch 3/100][Batch 1500/5005]	 Loss 5.054, Top 1-error 58.238, Top 5-error 32.346
Train with distillation: [Epoch 3/100][Batch 2000/5005]	 Loss 5.035, Top 1-error 58.120, Top 5-error 32.209
Train with distillation: [Epoch 3/100][Batch 2500/5005]	 Loss 5.016, Top 1-error 57.942, Top 5-error 32.065
Train with distillation: [Epoch 3/100][Batch 3000/5005]	 Loss 4.995, Top 1-error 57.765, Top 5-error 31.933
Train with distillation: [Epoch 3/100][Batch 3500/5005]	 Loss 4.975, Top 1-error 57.598, Top 5-error 31.780
Train with distillation: [Epoch 3/100][Batch 4000/5005]	 Loss 4.957, Top 1-error 57.477, Top 5-error 31.682
Train with distillation: [Epoch 3/100][Batch 4500/5005]	 Loss 4.938, Top 1-error 57.321, Top 5-error 31.561
Train with distillation: [Epoch 3/100][Batch 5000/5005]	 Loss 4.920, Top 1-error 57.198, Top 5-error 31.445
Train 	 Time Taken: 3181.07 sec
Test (on val set): [Epoch 3/100][Batch 0/196]	Time 1.902 (1.902)	Loss 1.2811 (1.2811)	Top 1-err 35.1562 (35.1562)	Top 5-err 10.9375 (10.9375)
* Epoch: [3/100]	 Top 1-err 53.410  Top 5-err 26.248	 Test Loss 2.318
Current best accuracy (top-1 and 5 error): 53.41 26.248
Train with distillation: [Epoch 4/100][Batch 0/5005]	 Loss 4.753, Top 1-error 53.906, Top 5-error 28.906
Train with distillation: [Epoch 4/100][Batch 500/5005]	 Loss 4.678, Top 1-error 55.396, Top 5-error 29.794
Train with distillation: [Epoch 4/100][Batch 1000/5005]	 Loss 4.677, Top 1-error 55.344, Top 5-error 29.779
Train with distillation: [Epoch 4/100][Batch 1500/5005]	 Loss 4.664, Top 1-error 55.199, Top 5-error 29.752
Train with distillation: [Epoch 4/100][Batch 2000/5005]	 Loss 4.654, Top 1-error 55.117, Top 5-error 29.649
Train with distillation: [Epoch 4/100][Batch 2500/5005]	 Loss 4.643, Top 1-error 55.052, Top 5-error 29.602
Train with distillation: [Epoch 4/100][Batch 3000/5005]	 Loss 4.633, Top 1-error 54.978, Top 5-error 29.525
Train with distillation: [Epoch 4/100][Batch 3500/5005]	 Loss 4.621, Top 1-error 54.841, Top 5-error 29.425
Train with distillation: [Epoch 4/100][Batch 4000/5005]	 Loss 4.611, Top 1-error 54.771, Top 5-error 29.370
Train with distillation: [Epoch 4/100][Batch 4500/5005]	 Loss 4.602, Top 1-error 54.695, Top 5-error 29.306
Train with distillation: [Epoch 4/100][Batch 5000/5005]	 Loss 4.595, Top 1-error 54.660, Top 5-error 29.253
Train 	 Time Taken: 3171.84 sec
Test (on val set): [Epoch 4/100][Batch 0/196]	Time 2.143 (2.143)	Loss 1.5194 (1.5194)	Top 1-err 40.2344 (40.2344)	Top 5-err 12.8906 (12.8906)
* Epoch: [4/100]	 Top 1-err 52.550  Top 5-err 25.808	 Test Loss 2.275
Current best accuracy (top-1 and 5 error): 52.55 25.808
Train with distillation: [Epoch 5/100][Batch 0/5005]	 Loss 4.470, Top 1-error 55.469, Top 5-error 30.078
Train with distillation: [Epoch 5/100][Batch 500/5005]	 Loss 4.426, Top 1-error 53.155, Top 5-error 28.000
Train with distillation: [Epoch 5/100][Batch 1000/5005]	 Loss 4.423, Top 1-error 53.123, Top 5-error 27.963
Train with distillation: [Epoch 5/100][Batch 1500/5005]	 Loss 4.427, Top 1-error 53.144, Top 5-error 28.049
Train with distillation: [Epoch 5/100][Batch 2000/5005]	 Loss 4.426, Top 1-error 53.179, Top 5-error 28.047
Train with distillation: [Epoch 5/100][Batch 2500/5005]	 Loss 4.420, Top 1-error 53.154, Top 5-error 27.985
Train with distillation: [Epoch 5/100][Batch 3000/5005]	 Loss 4.417, Top 1-error 53.148, Top 5-error 27.968
Train with distillation: [Epoch 5/100][Batch 3500/5005]	 Loss 4.413, Top 1-error 53.131, Top 5-error 27.965
Train with distillation: [Epoch 5/100][Batch 4000/5005]	 Loss 4.408, Top 1-error 53.102, Top 5-error 27.931
Train with distillation: [Epoch 5/100][Batch 4500/5005]	 Loss 4.403, Top 1-error 53.071, Top 5-error 27.896
Train with distillation: [Epoch 5/100][Batch 5000/5005]	 Loss 4.398, Top 1-error 53.038, Top 5-error 27.880
Train 	 Time Taken: 3165.60 sec
Test (on val set): [Epoch 5/100][Batch 0/196]	Time 1.924 (1.924)	Loss 1.6275 (1.6275)	Top 1-err 41.7969 (41.7969)	Top 5-err 15.2344 (15.2344)
* Epoch: [5/100]	 Top 1-err 50.424  Top 5-err 24.116	 Test Loss 2.171
Current best accuracy (top-1 and 5 error): 50.424 24.116
Train with distillation: [Epoch 6/100][Batch 0/5005]	 Loss 4.230, Top 1-error 48.438, Top 5-error 27.344
Train with distillation: [Epoch 6/100][Batch 500/5005]	 Loss 4.290, Top 1-error 52.052, Top 5-error 27.001
Train with distillation: [Epoch 6/100][Batch 1000/5005]	 Loss 4.286, Top 1-error 51.999, Top 5-error 27.037
Train with distillation: [Epoch 6/100][Batch 1500/5005]	 Loss 4.285, Top 1-error 52.062, Top 5-error 27.061
Train with distillation: [Epoch 6/100][Batch 2000/5005]	 Loss 4.287, Top 1-error 52.069, Top 5-error 27.092
Train with distillation: [Epoch 6/100][Batch 2500/5005]	 Loss 4.285, Top 1-error 52.048, Top 5-error 27.079
Train with distillation: [Epoch 6/100][Batch 3000/5005]	 Loss 4.283, Top 1-error 52.034, Top 5-error 27.064
Train with distillation: [Epoch 6/100][Batch 3500/5005]	 Loss 4.281, Top 1-error 52.041, Top 5-error 27.054
Train with distillation: [Epoch 6/100][Batch 4000/5005]	 Loss 4.280, Top 1-error 52.063, Top 5-error 27.047
Train with distillation: [Epoch 6/100][Batch 4500/5005]	 Loss 4.277, Top 1-error 52.051, Top 5-error 27.044
Train with distillation: [Epoch 6/100][Batch 5000/5005]	 Loss 4.274, Top 1-error 52.048, Top 5-error 27.017
Train 	 Time Taken: 3156.02 sec
Test (on val set): [Epoch 6/100][Batch 0/196]	Time 2.060 (2.060)	Loss 1.2523 (1.2523)	Top 1-err 35.9375 (35.9375)	Top 5-err 10.5469 (10.5469)
* Epoch: [6/100]	 Top 1-err 50.016  Top 5-err 23.708	 Test Loss 2.154
Current best accuracy (top-1 and 5 error): 50.016 23.708
Train with distillation: [Epoch 7/100][Batch 0/5005]	 Loss 4.029, Top 1-error 48.047, Top 5-error 25.000
Train with distillation: [Epoch 7/100][Batch 500/5005]	 Loss 4.203, Top 1-error 51.220, Top 5-error 26.410
Train with distillation: [Epoch 7/100][Batch 1000/5005]	 Loss 4.203, Top 1-error 51.351, Top 5-error 26.435
Train with distillation: [Epoch 7/100][Batch 1500/5005]	 Loss 4.206, Top 1-error 51.498, Top 5-error 26.519
Train with distillation: [Epoch 7/100][Batch 2000/5005]	 Loss 4.202, Top 1-error 51.460, Top 5-error 26.444
Train with distillation: [Epoch 7/100][Batch 2500/5005]	 Loss 4.202, Top 1-error 51.453, Top 5-error 26.426
Train with distillation: [Epoch 7/100][Batch 3000/5005]	 Loss 4.200, Top 1-error 51.463, Top 5-error 26.467
Train with distillation: [Epoch 7/100][Batch 3500/5005]	 Loss 4.196, Top 1-error 51.451, Top 5-error 26.448
Train with distillation: [Epoch 7/100][Batch 4000/5005]	 Loss 4.193, Top 1-error 51.427, Top 5-error 26.438
Train with distillation: [Epoch 7/100][Batch 4500/5005]	 Loss 4.192, Top 1-error 51.417, Top 5-error 26.442
Train with distillation: [Epoch 7/100][Batch 5000/5005]	 Loss 4.189, Top 1-error 51.393, Top 5-error 26.441
Train 	 Time Taken: 3104.71 sec
Test (on val set): [Epoch 7/100][Batch 0/196]	Time 1.848 (1.848)	Loss 1.2491 (1.2491)	Top 1-err 35.1562 (35.1562)	Top 5-err 9.3750 (9.3750)
* Epoch: [7/100]	 Top 1-err 49.136  Top 5-err 23.174	 Test Loss 2.108
Current best accuracy (top-1 and 5 error): 49.136 23.174
Train with distillation: [Epoch 8/100][Batch 0/5005]	 Loss 3.961, Top 1-error 48.047, Top 5-error 22.656
Train with distillation: [Epoch 8/100][Batch 500/5005]	 Loss 4.095, Top 1-error 50.231, Top 5-error 25.646
Train with distillation: [Epoch 8/100][Batch 1000/5005]	 Loss 4.104, Top 1-error 50.512, Top 5-error 25.736
Train with distillation: [Epoch 8/100][Batch 1500/5005]	 Loss 4.112, Top 1-error 50.621, Top 5-error 25.765
Train with distillation: [Epoch 8/100][Batch 2000/5005]	 Loss 4.117, Top 1-error 50.682, Top 5-error 25.870
Train with distillation: [Epoch 8/100][Batch 2500/5005]	 Loss 4.117, Top 1-error 50.680, Top 5-error 25.886
Train with distillation: [Epoch 8/100][Batch 3000/5005]	 Loss 4.119, Top 1-error 50.722, Top 5-error 25.918
Train with distillation: [Epoch 8/100][Batch 3500/5005]	 Loss 4.119, Top 1-error 50.743, Top 5-error 25.933
Train with distillation: [Epoch 8/100][Batch 4000/5005]	 Loss 4.119, Top 1-error 50.754, Top 5-error 25.944
Train with distillation: [Epoch 8/100][Batch 4500/5005]	 Loss 4.118, Top 1-error 50.758, Top 5-error 25.949
Train with distillation: [Epoch 8/100][Batch 5000/5005]	 Loss 4.117, Top 1-error 50.771, Top 5-error 25.954
Train 	 Time Taken: 3092.24 sec
Test (on val set): [Epoch 8/100][Batch 0/196]	Time 1.939 (1.939)	Loss 1.6417 (1.6417)	Top 1-err 40.6250 (40.6250)	Top 5-err 13.6719 (13.6719)
* Epoch: [8/100]	 Top 1-err 50.250  Top 5-err 24.192	 Test Loss 2.170
Current best accuracy (top-1 and 5 error): 49.136 23.174
Train with distillation: [Epoch 9/100][Batch 0/5005]	 Loss 4.189, Top 1-error 49.219, Top 5-error 28.906
Train with distillation: [Epoch 9/100][Batch 500/5005]	 Loss 4.044, Top 1-error 49.904, Top 5-error 25.327
Train with distillation: [Epoch 9/100][Batch 1000/5005]	 Loss 4.064, Top 1-error 50.117, Top 5-error 25.504
Train with distillation: [Epoch 9/100][Batch 1500/5005]	 Loss 4.069, Top 1-error 50.265, Top 5-error 25.569
Train with distillation: [Epoch 9/100][Batch 2000/5005]	 Loss 4.069, Top 1-error 50.284, Top 5-error 25.612
Train with distillation: [Epoch 9/100][Batch 2500/5005]	 Loss 4.072, Top 1-error 50.381, Top 5-error 25.648
Train with distillation: [Epoch 9/100][Batch 3000/5005]	 Loss 4.070, Top 1-error 50.369, Top 5-error 25.632
Train with distillation: [Epoch 9/100][Batch 3500/5005]	 Loss 4.070, Top 1-error 50.383, Top 5-error 25.624
Train with distillation: [Epoch 9/100][Batch 4000/5005]	 Loss 4.070, Top 1-error 50.414, Top 5-error 25.624
Train with distillation: [Epoch 9/100][Batch 4500/5005]	 Loss 4.070, Top 1-error 50.414, Top 5-error 25.648
Train with distillation: [Epoch 9/100][Batch 5000/5005]	 Loss 4.069, Top 1-error 50.432, Top 5-error 25.654
Train 	 Time Taken: 3072.94 sec
Test (on val set): [Epoch 9/100][Batch 0/196]	Time 1.830 (1.830)	Loss 1.0833 (1.0833)	Top 1-err 27.7344 (27.7344)	Top 5-err 9.7656 (9.7656)
* Epoch: [9/100]	 Top 1-err 48.514  Top 5-err 22.534	 Test Loss 2.107
Current best accuracy (top-1 and 5 error): 48.514 22.534
Train with distillation: [Epoch 10/100][Batch 0/5005]	 Loss 3.923, Top 1-error 51.953, Top 5-error 24.219
Train with distillation: [Epoch 10/100][Batch 500/5005]	 Loss 4.001, Top 1-error 49.603, Top 5-error 25.022
Train with distillation: [Epoch 10/100][Batch 1000/5005]	 Loss 4.010, Top 1-error 49.694, Top 5-error 25.098
Train with distillation: [Epoch 10/100][Batch 1500/5005]	 Loss 4.015, Top 1-error 49.778, Top 5-error 25.191
Train with distillation: [Epoch 10/100][Batch 2000/5005]	 Loss 4.015, Top 1-error 49.828, Top 5-error 25.223
Train with distillation: [Epoch 10/100][Batch 2500/5005]	 Loss 4.016, Top 1-error 49.886, Top 5-error 25.254
Train with distillation: [Epoch 10/100][Batch 3000/5005]	 Loss 4.019, Top 1-error 49.956, Top 5-error 25.295
Train with distillation: [Epoch 10/100][Batch 3500/5005]	 Loss 4.023, Top 1-error 50.009, Top 5-error 25.353
Train with distillation: [Epoch 10/100][Batch 4000/5005]	 Loss 4.023, Top 1-error 50.039, Top 5-error 25.351
Train with distillation: [Epoch 10/100][Batch 4500/5005]	 Loss 4.024, Top 1-error 50.060, Top 5-error 25.344
Train with distillation: [Epoch 10/100][Batch 5000/5005]	 Loss 4.023, Top 1-error 50.062, Top 5-error 25.347
Train 	 Time Taken: 3066.17 sec
Test (on val set): [Epoch 10/100][Batch 0/196]	Time 1.943 (1.943)	Loss 1.1508 (1.1508)	Top 1-err 33.2031 (33.2031)	Top 5-err 9.7656 (9.7656)
* Epoch: [10/100]	 Top 1-err 48.816  Top 5-err 22.912	 Test Loss 2.092
Current best accuracy (top-1 and 5 error): 48.514 22.534
Train with distillation: [Epoch 11/100][Batch 0/5005]	 Loss 3.907, Top 1-error 45.703, Top 5-error 25.000
Train with distillation: [Epoch 11/100][Batch 500/5005]	 Loss 3.950, Top 1-error 49.204, Top 5-error 24.722
Train with distillation: [Epoch 11/100][Batch 1000/5005]	 Loss 3.968, Top 1-error 49.519, Top 5-error 24.943
Train with distillation: [Epoch 11/100][Batch 1500/5005]	 Loss 3.975, Top 1-error 49.591, Top 5-error 24.995
Train with distillation: [Epoch 11/100][Batch 2000/5005]	 Loss 3.980, Top 1-error 49.632, Top 5-error 25.050
Train with distillation: [Epoch 11/100][Batch 2500/5005]	 Loss 3.982, Top 1-error 49.641, Top 5-error 25.048
Train with distillation: [Epoch 11/100][Batch 3000/5005]	 Loss 3.985, Top 1-error 49.679, Top 5-error 25.088
Train with distillation: [Epoch 11/100][Batch 3500/5005]	 Loss 3.985, Top 1-error 49.672, Top 5-error 25.051
Train with distillation: [Epoch 11/100][Batch 4000/5005]	 Loss 3.985, Top 1-error 49.683, Top 5-error 25.058
Train with distillation: [Epoch 11/100][Batch 4500/5005]	 Loss 3.986, Top 1-error 49.707, Top 5-error 25.060
Train with distillation: [Epoch 11/100][Batch 5000/5005]	 Loss 3.985, Top 1-error 49.685, Top 5-error 25.057
Train 	 Time Taken: 3058.74 sec
Test (on val set): [Epoch 11/100][Batch 0/196]	Time 1.778 (1.778)	Loss 1.2543 (1.2543)	Top 1-err 30.0781 (30.0781)	Top 5-err 9.7656 (9.7656)
* Epoch: [11/100]	 Top 1-err 46.990  Top 5-err 21.002	 Test Loss 1.989
Current best accuracy (top-1 and 5 error): 46.99 21.002
Train with distillation: [Epoch 12/100][Batch 0/5005]	 Loss 4.226, Top 1-error 51.172, Top 5-error 25.000
Train with distillation: [Epoch 12/100][Batch 500/5005]	 Loss 3.927, Top 1-error 49.057, Top 5-error 24.645
Train with distillation: [Epoch 12/100][Batch 1000/5005]	 Loss 3.936, Top 1-error 49.196, Top 5-error 24.660
Train with distillation: [Epoch 12/100][Batch 1500/5005]	 Loss 3.942, Top 1-error 49.274, Top 5-error 24.748
Train with distillation: [Epoch 12/100][Batch 2000/5005]	 Loss 3.946, Top 1-error 49.351, Top 5-error 24.816
Train with distillation: [Epoch 12/100][Batch 2500/5005]	 Loss 3.949, Top 1-error 49.374, Top 5-error 24.835
Train with distillation: [Epoch 12/100][Batch 3000/5005]	 Loss 3.951, Top 1-error 49.431, Top 5-error 24.856
Train with distillation: [Epoch 12/100][Batch 3500/5005]	 Loss 3.954, Top 1-error 49.469, Top 5-error 24.903
Train with distillation: [Epoch 12/100][Batch 4000/5005]	 Loss 3.954, Top 1-error 49.461, Top 5-error 24.897
Train with distillation: [Epoch 12/100][Batch 4500/5005]	 Loss 3.958, Top 1-error 49.522, Top 5-error 24.943
Train with distillation: [Epoch 12/100][Batch 5000/5005]	 Loss 3.958, Top 1-error 49.535, Top 5-error 24.948
Train 	 Time Taken: 3044.06 sec
Test (on val set): [Epoch 12/100][Batch 0/196]	Time 1.871 (1.871)	Loss 1.3345 (1.3345)	Top 1-err 34.3750 (34.3750)	Top 5-err 12.1094 (12.1094)
* Epoch: [12/100]	 Top 1-err 47.314  Top 5-err 21.166	 Test Loss 2.006
Current best accuracy (top-1 and 5 error): 46.99 21.002
Train with distillation: [Epoch 13/100][Batch 0/5005]	 Loss 3.897, Top 1-error 48.828, Top 5-error 25.391
Train with distillation: [Epoch 13/100][Batch 500/5005]	 Loss 3.910, Top 1-error 49.036, Top 5-error 24.483
Train with distillation: [Epoch 13/100][Batch 1000/5005]	 Loss 3.912, Top 1-error 49.074, Top 5-error 24.433
Train with distillation: [Epoch 13/100][Batch 1500/5005]	 Loss 3.920, Top 1-error 49.167, Top 5-error 24.556
Train with distillation: [Epoch 13/100][Batch 2000/5005]	 Loss 3.924, Top 1-error 49.198, Top 5-error 24.590
Train with distillation: [Epoch 13/100][Batch 2500/5005]	 Loss 3.928, Top 1-error 49.259, Top 5-error 24.667
Train with distillation: [Epoch 13/100][Batch 3000/5005]	 Loss 3.931, Top 1-error 49.325, Top 5-error 24.726
Train with distillation: [Epoch 13/100][Batch 3500/5005]	 Loss 3.930, Top 1-error 49.302, Top 5-error 24.726
Train with distillation: [Epoch 13/100][Batch 4000/5005]	 Loss 3.932, Top 1-error 49.307, Top 5-error 24.749
Train with distillation: [Epoch 13/100][Batch 4500/5005]	 Loss 3.931, Top 1-error 49.283, Top 5-error 24.753
Train with distillation: [Epoch 13/100][Batch 5000/5005]	 Loss 3.933, Top 1-error 49.283, Top 5-error 24.766
Train 	 Time Taken: 3029.41 sec
Test (on val set): [Epoch 13/100][Batch 0/196]	Time 1.855 (1.855)	Loss 1.2185 (1.2185)	Top 1-err 31.2500 (31.2500)	Top 5-err 10.9375 (10.9375)
* Epoch: [13/100]	 Top 1-err 46.510  Top 5-err 20.590	 Test Loss 1.961
Current best accuracy (top-1 and 5 error): 46.51 20.59
Train with distillation: [Epoch 14/100][Batch 0/5005]	 Loss 3.648, Top 1-error 44.922, Top 5-error 20.312
Train with distillation: [Epoch 14/100][Batch 500/5005]	 Loss 3.879, Top 1-error 48.650, Top 5-error 24.218
Train with distillation: [Epoch 14/100][Batch 1000/5005]	 Loss 3.893, Top 1-error 48.848, Top 5-error 24.380
Train with distillation: [Epoch 14/100][Batch 1500/5005]	 Loss 3.896, Top 1-error 48.857, Top 5-error 24.436
Train with distillation: [Epoch 14/100][Batch 2000/5005]	 Loss 3.899, Top 1-error 48.924, Top 5-error 24.510
Train with distillation: [Epoch 14/100][Batch 2500/5005]	 Loss 3.904, Top 1-error 49.000, Top 5-error 24.560
Train with distillation: [Epoch 14/100][Batch 3000/5005]	 Loss 3.906, Top 1-error 49.031, Top 5-error 24.581
Train with distillation: [Epoch 14/100][Batch 3500/5005]	 Loss 3.908, Top 1-error 49.063, Top 5-error 24.599
Train with distillation: [Epoch 14/100][Batch 4000/5005]	 Loss 3.909, Top 1-error 49.067, Top 5-error 24.603
Train with distillation: [Epoch 14/100][Batch 4500/5005]	 Loss 3.910, Top 1-error 49.079, Top 5-error 24.608
Train with distillation: [Epoch 14/100][Batch 5000/5005]	 Loss 3.911, Top 1-error 49.090, Top 5-error 24.613
Train 	 Time Taken: 3044.54 sec
Test (on val set): [Epoch 14/100][Batch 0/196]	Time 1.853 (1.853)	Loss 1.3929 (1.3929)	Top 1-err 37.8906 (37.8906)	Top 5-err 10.1562 (10.1562)
* Epoch: [14/100]	 Top 1-err 47.520  Top 5-err 21.838	 Test Loss 2.032
Current best accuracy (top-1 and 5 error): 46.51 20.59
Train with distillation: [Epoch 15/100][Batch 0/5005]	 Loss 4.021, Top 1-error 46.875, Top 5-error 28.125
Train with distillation: [Epoch 15/100][Batch 500/5005]	 Loss 3.872, Top 1-error 48.727, Top 5-error 24.220
Train with distillation: [Epoch 15/100][Batch 1000/5005]	 Loss 3.876, Top 1-error 48.857, Top 5-error 24.303
Train with distillation: [Epoch 15/100][Batch 1500/5005]	 Loss 3.876, Top 1-error 48.868, Top 5-error 24.283
Train with distillation: [Epoch 15/100][Batch 2000/5005]	 Loss 3.882, Top 1-error 48.936, Top 5-error 24.379
Train with distillation: [Epoch 15/100][Batch 2500/5005]	 Loss 3.884, Top 1-error 48.921, Top 5-error 24.417
Train with distillation: [Epoch 15/100][Batch 3000/5005]	 Loss 3.886, Top 1-error 48.927, Top 5-error 24.434
Train with distillation: [Epoch 15/100][Batch 3500/5005]	 Loss 3.888, Top 1-error 48.981, Top 5-error 24.465
Train with distillation: [Epoch 15/100][Batch 4000/5005]	 Loss 3.890, Top 1-error 49.014, Top 5-error 24.492
Train with distillation: [Epoch 15/100][Batch 4500/5005]	 Loss 3.891, Top 1-error 49.019, Top 5-error 24.497
Train with distillation: [Epoch 15/100][Batch 5000/5005]	 Loss 3.892, Top 1-error 49.024, Top 5-error 24.517
Train 	 Time Taken: 3041.26 sec
Test (on val set): [Epoch 15/100][Batch 0/196]	Time 1.854 (1.854)	Loss 1.2259 (1.2259)	Top 1-err 30.0781 (30.0781)	Top 5-err 11.3281 (11.3281)
* Epoch: [15/100]	 Top 1-err 46.744  Top 5-err 20.998	 Test Loss 1.984
Current best accuracy (top-1 and 5 error): 46.51 20.59
Train with distillation: [Epoch 16/100][Batch 0/5005]	 Loss 3.954, Top 1-error 51.562, Top 5-error 28.125
Train with distillation: [Epoch 16/100][Batch 500/5005]	 Loss 3.847, Top 1-error 48.557, Top 5-error 24.092
Train with distillation: [Epoch 16/100][Batch 1000/5005]	 Loss 3.842, Top 1-error 48.478, Top 5-error 24.004
Train with distillation: [Epoch 16/100][Batch 1500/5005]	 Loss 3.854, Top 1-error 48.626, Top 5-error 24.119
Train with distillation: [Epoch 16/100][Batch 2000/5005]	 Loss 3.860, Top 1-error 48.658, Top 5-error 24.140
Train with distillation: [Epoch 16/100][Batch 2500/5005]	 Loss 3.864, Top 1-error 48.706, Top 5-error 24.175
Train with distillation: [Epoch 16/100][Batch 3000/5005]	 Loss 3.867, Top 1-error 48.757, Top 5-error 24.221
Train with distillation: [Epoch 16/100][Batch 3500/5005]	 Loss 3.870, Top 1-error 48.792, Top 5-error 24.260
Train with distillation: [Epoch 16/100][Batch 4000/5005]	 Loss 3.871, Top 1-error 48.796, Top 5-error 24.282
Train with distillation: [Epoch 16/100][Batch 4500/5005]	 Loss 3.871, Top 1-error 48.801, Top 5-error 24.302
Train with distillation: [Epoch 16/100][Batch 5000/5005]	 Loss 3.872, Top 1-error 48.818, Top 5-error 24.313
Train 	 Time Taken: 3033.82 sec
Test (on val set): [Epoch 16/100][Batch 0/196]	Time 1.895 (1.895)	Loss 1.0084 (1.0084)	Top 1-err 28.5156 (28.5156)	Top 5-err 7.4219 (7.4219)
* Epoch: [16/100]	 Top 1-err 46.816  Top 5-err 21.444	 Test Loss 1.997
Current best accuracy (top-1 and 5 error): 46.51 20.59
Train with distillation: [Epoch 17/100][Batch 0/5005]	 Loss 3.629, Top 1-error 41.016, Top 5-error 20.703
Train with distillation: [Epoch 17/100][Batch 500/5005]	 Loss 3.830, Top 1-error 48.204, Top 5-error 23.962
Train with distillation: [Epoch 17/100][Batch 1000/5005]	 Loss 3.840, Top 1-error 48.432, Top 5-error 24.157
Train with distillation: [Epoch 17/100][Batch 1500/5005]	 Loss 3.846, Top 1-error 48.541, Top 5-error 24.127
Train with distillation: [Epoch 17/100][Batch 2000/5005]	 Loss 3.847, Top 1-error 48.541, Top 5-error 24.162
Train with distillation: [Epoch 17/100][Batch 2500/5005]	 Loss 3.850, Top 1-error 48.613, Top 5-error 24.180
Train with distillation: [Epoch 17/100][Batch 3000/5005]	 Loss 3.851, Top 1-error 48.621, Top 5-error 24.197
Train with distillation: [Epoch 17/100][Batch 3500/5005]	 Loss 3.855, Top 1-error 48.668, Top 5-error 24.229
Train with distillation: [Epoch 17/100][Batch 4000/5005]	 Loss 3.854, Top 1-error 48.664, Top 5-error 24.217
Train with distillation: [Epoch 17/100][Batch 4500/5005]	 Loss 3.854, Top 1-error 48.664, Top 5-error 24.224
Train with distillation: [Epoch 17/100][Batch 5000/5005]	 Loss 3.856, Top 1-error 48.689, Top 5-error 24.239
Train 	 Time Taken: 3018.52 sec
Test (on val set): [Epoch 17/100][Batch 0/196]	Time 1.847 (1.847)	Loss 1.4277 (1.4277)	Top 1-err 39.0625 (39.0625)	Top 5-err 12.1094 (12.1094)
* Epoch: [17/100]	 Top 1-err 47.480  Top 5-err 22.100	 Test Loss 2.037
Current best accuracy (top-1 and 5 error): 46.51 20.59
Train with distillation: [Epoch 18/100][Batch 0/5005]	 Loss 3.994, Top 1-error 46.875, Top 5-error 26.172
Train with distillation: [Epoch 18/100][Batch 500/5005]	 Loss 3.812, Top 1-error 48.283, Top 5-error 23.855
Train with distillation: [Epoch 18/100][Batch 1000/5005]	 Loss 3.826, Top 1-error 48.441, Top 5-error 24.013
Train with distillation: [Epoch 18/100][Batch 1500/5005]	 Loss 3.829, Top 1-error 48.483, Top 5-error 23.994
Train with distillation: [Epoch 18/100][Batch 2000/5005]	 Loss 3.832, Top 1-error 48.538, Top 5-error 24.036
Train with distillation: [Epoch 18/100][Batch 2500/5005]	 Loss 3.833, Top 1-error 48.528, Top 5-error 23.979
Train with distillation: [Epoch 18/100][Batch 3000/5005]	 Loss 3.836, Top 1-error 48.541, Top 5-error 24.030
Train with distillation: [Epoch 18/100][Batch 3500/5005]	 Loss 3.837, Top 1-error 48.544, Top 5-error 24.041
Train with distillation: [Epoch 18/100][Batch 4000/5005]	 Loss 3.839, Top 1-error 48.581, Top 5-error 24.084
Train with distillation: [Epoch 18/100][Batch 4500/5005]	 Loss 3.841, Top 1-error 48.608, Top 5-error 24.095
Train with distillation: [Epoch 18/100][Batch 5000/5005]	 Loss 3.841, Top 1-error 48.626, Top 5-error 24.106
Train 	 Time Taken: 3005.98 sec
Test (on val set): [Epoch 18/100][Batch 0/196]	Time 1.836 (1.836)	Loss 1.1425 (1.1425)	Top 1-err 30.8594 (30.8594)	Top 5-err 8.5938 (8.5938)
* Epoch: [18/100]	 Top 1-err 45.416  Top 5-err 19.900	 Test Loss 1.909
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 19/100][Batch 0/5005]	 Loss 3.882, Top 1-error 51.953, Top 5-error 24.219
Train with distillation: [Epoch 19/100][Batch 500/5005]	 Loss 3.794, Top 1-error 48.039, Top 5-error 23.672
Train with distillation: [Epoch 19/100][Batch 1000/5005]	 Loss 3.802, Top 1-error 48.089, Top 5-error 23.754
Train with distillation: [Epoch 19/100][Batch 1500/5005]	 Loss 3.816, Top 1-error 48.245, Top 5-error 23.916
Train with distillation: [Epoch 19/100][Batch 2000/5005]	 Loss 3.823, Top 1-error 48.338, Top 5-error 23.971
Train with distillation: [Epoch 19/100][Batch 2500/5005]	 Loss 3.824, Top 1-error 48.392, Top 5-error 23.993
Train with distillation: [Epoch 19/100][Batch 3000/5005]	 Loss 3.825, Top 1-error 48.427, Top 5-error 24.026
Train with distillation: [Epoch 19/100][Batch 3500/5005]	 Loss 3.828, Top 1-error 48.429, Top 5-error 24.037
Train with distillation: [Epoch 19/100][Batch 4000/5005]	 Loss 3.832, Top 1-error 48.488, Top 5-error 24.073
Train with distillation: [Epoch 19/100][Batch 4500/5005]	 Loss 3.832, Top 1-error 48.497, Top 5-error 24.081
Train with distillation: [Epoch 19/100][Batch 5000/5005]	 Loss 3.834, Top 1-error 48.517, Top 5-error 24.102
Train 	 Time Taken: 2991.60 sec
Test (on val set): [Epoch 19/100][Batch 0/196]	Time 1.838 (1.838)	Loss 1.3955 (1.3955)	Top 1-err 35.9375 (35.9375)	Top 5-err 12.1094 (12.1094)
* Epoch: [19/100]	 Top 1-err 46.296  Top 5-err 21.084	 Test Loss 1.967
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 20/100][Batch 0/5005]	 Loss 3.569, Top 1-error 42.188, Top 5-error 19.531
Train with distillation: [Epoch 20/100][Batch 500/5005]	 Loss 3.783, Top 1-error 47.947, Top 5-error 23.600
Train with distillation: [Epoch 20/100][Batch 1000/5005]	 Loss 3.797, Top 1-error 48.100, Top 5-error 23.709
Train with distillation: [Epoch 20/100][Batch 1500/5005]	 Loss 3.802, Top 1-error 48.110, Top 5-error 23.778
Train with distillation: [Epoch 20/100][Batch 2000/5005]	 Loss 3.805, Top 1-error 48.124, Top 5-error 23.817
Train with distillation: [Epoch 20/100][Batch 2500/5005]	 Loss 3.811, Top 1-error 48.197, Top 5-error 23.872
Train with distillation: [Epoch 20/100][Batch 3000/5005]	 Loss 3.815, Top 1-error 48.236, Top 5-error 23.929
Train with distillation: [Epoch 20/100][Batch 3500/5005]	 Loss 3.817, Top 1-error 48.306, Top 5-error 23.949
Train with distillation: [Epoch 20/100][Batch 4000/5005]	 Loss 3.819, Top 1-error 48.351, Top 5-error 23.975
Train with distillation: [Epoch 20/100][Batch 4500/5005]	 Loss 3.820, Top 1-error 48.367, Top 5-error 23.985
Train with distillation: [Epoch 20/100][Batch 5000/5005]	 Loss 3.820, Top 1-error 48.373, Top 5-error 23.999
Train 	 Time Taken: 2980.40 sec
Test (on val set): [Epoch 20/100][Batch 0/196]	Time 1.874 (1.874)	Loss 1.2281 (1.2281)	Top 1-err 30.8594 (30.8594)	Top 5-err 9.7656 (9.7656)
* Epoch: [20/100]	 Top 1-err 45.998  Top 5-err 19.976	 Test Loss 1.918
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 21/100][Batch 0/5005]	 Loss 3.851, Top 1-error 50.391, Top 5-error 26.562
Train with distillation: [Epoch 21/100][Batch 500/5005]	 Loss 3.777, Top 1-error 48.036, Top 5-error 23.582
Train with distillation: [Epoch 21/100][Batch 1000/5005]	 Loss 3.786, Top 1-error 48.054, Top 5-error 23.696
Train with distillation: [Epoch 21/100][Batch 1500/5005]	 Loss 3.795, Top 1-error 48.165, Top 5-error 23.804
Train with distillation: [Epoch 21/100][Batch 2000/5005]	 Loss 3.797, Top 1-error 48.191, Top 5-error 23.826
Train with distillation: [Epoch 21/100][Batch 2500/5005]	 Loss 3.803, Top 1-error 48.229, Top 5-error 23.893
Train with distillation: [Epoch 21/100][Batch 3000/5005]	 Loss 3.806, Top 1-error 48.271, Top 5-error 23.934
Train with distillation: [Epoch 21/100][Batch 3500/5005]	 Loss 3.807, Top 1-error 48.281, Top 5-error 23.936
Train with distillation: [Epoch 21/100][Batch 4000/5005]	 Loss 3.810, Top 1-error 48.313, Top 5-error 23.960
Train with distillation: [Epoch 21/100][Batch 4500/5005]	 Loss 3.811, Top 1-error 48.334, Top 5-error 23.960
Train with distillation: [Epoch 21/100][Batch 5000/5005]	 Loss 3.812, Top 1-error 48.355, Top 5-error 23.983
Train 	 Time Taken: 2978.39 sec
Test (on val set): [Epoch 21/100][Batch 0/196]	Time 1.846 (1.846)	Loss 1.1233 (1.1233)	Top 1-err 31.6406 (31.6406)	Top 5-err 7.0312 (7.0312)
* Epoch: [21/100]	 Top 1-err 46.116  Top 5-err 20.444	 Test Loss 1.958
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 22/100][Batch 0/5005]	 Loss 3.951, Top 1-error 46.875, Top 5-error 25.781
Train with distillation: [Epoch 22/100][Batch 500/5005]	 Loss 3.751, Top 1-error 47.691, Top 5-error 23.266
Train with distillation: [Epoch 22/100][Batch 1000/5005]	 Loss 3.769, Top 1-error 47.864, Top 5-error 23.503
Train with distillation: [Epoch 22/100][Batch 1500/5005]	 Loss 3.775, Top 1-error 47.966, Top 5-error 23.573
Train with distillation: [Epoch 22/100][Batch 2000/5005]	 Loss 3.779, Top 1-error 48.015, Top 5-error 23.622
Train with distillation: [Epoch 22/100][Batch 2500/5005]	 Loss 3.787, Top 1-error 48.100, Top 5-error 23.737
Train with distillation: [Epoch 22/100][Batch 3000/5005]	 Loss 3.788, Top 1-error 48.079, Top 5-error 23.741
Train with distillation: [Epoch 22/100][Batch 3500/5005]	 Loss 3.792, Top 1-error 48.129, Top 5-error 23.788
Train with distillation: [Epoch 22/100][Batch 4000/5005]	 Loss 3.794, Top 1-error 48.171, Top 5-error 23.814
Train with distillation: [Epoch 22/100][Batch 4500/5005]	 Loss 3.798, Top 1-error 48.200, Top 5-error 23.839
Train with distillation: [Epoch 22/100][Batch 5000/5005]	 Loss 3.800, Top 1-error 48.251, Top 5-error 23.875
Train 	 Time Taken: 2973.54 sec
Test (on val set): [Epoch 22/100][Batch 0/196]	Time 1.843 (1.843)	Loss 1.3141 (1.3141)	Top 1-err 40.6250 (40.6250)	Top 5-err 7.4219 (7.4219)
* Epoch: [22/100]	 Top 1-err 46.154  Top 5-err 20.540	 Test Loss 1.938
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 23/100][Batch 0/5005]	 Loss 3.848, Top 1-error 51.953, Top 5-error 21.484
Train with distillation: [Epoch 23/100][Batch 500/5005]	 Loss 3.753, Top 1-error 47.425, Top 5-error 23.676
Train with distillation: [Epoch 23/100][Batch 1000/5005]	 Loss 3.765, Top 1-error 47.652, Top 5-error 23.641
Train with distillation: [Epoch 23/100][Batch 1500/5005]	 Loss 3.773, Top 1-error 47.780, Top 5-error 23.670
Train with distillation: [Epoch 23/100][Batch 2000/5005]	 Loss 3.778, Top 1-error 47.842, Top 5-error 23.697
Train with distillation: [Epoch 23/100][Batch 2500/5005]	 Loss 3.782, Top 1-error 47.944, Top 5-error 23.733
Train with distillation: [Epoch 23/100][Batch 3000/5005]	 Loss 3.786, Top 1-error 48.018, Top 5-error 23.789
Train with distillation: [Epoch 23/100][Batch 3500/5005]	 Loss 3.788, Top 1-error 48.107, Top 5-error 23.812
Train with distillation: [Epoch 23/100][Batch 4000/5005]	 Loss 3.789, Top 1-error 48.146, Top 5-error 23.815
Train with distillation: [Epoch 23/100][Batch 4500/5005]	 Loss 3.790, Top 1-error 48.149, Top 5-error 23.804
Train with distillation: [Epoch 23/100][Batch 5000/5005]	 Loss 3.791, Top 1-error 48.185, Top 5-error 23.831
Train 	 Time Taken: 2959.85 sec
Test (on val set): [Epoch 23/100][Batch 0/196]	Time 1.779 (1.779)	Loss 0.9797 (0.9797)	Top 1-err 26.1719 (26.1719)	Top 5-err 9.3750 (9.3750)
* Epoch: [23/100]	 Top 1-err 46.040  Top 5-err 20.502	 Test Loss 1.947
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 24/100][Batch 0/5005]	 Loss 3.911, Top 1-error 46.875, Top 5-error 21.875
Train with distillation: [Epoch 24/100][Batch 500/5005]	 Loss 3.746, Top 1-error 47.620, Top 5-error 23.495
Train with distillation: [Epoch 24/100][Batch 1000/5005]	 Loss 3.757, Top 1-error 47.740, Top 5-error 23.651
Train with distillation: [Epoch 24/100][Batch 1500/5005]	 Loss 3.762, Top 1-error 47.811, Top 5-error 23.583
Train with distillation: [Epoch 24/100][Batch 2000/5005]	 Loss 3.766, Top 1-error 47.829, Top 5-error 23.625
Train with distillation: [Epoch 24/100][Batch 2500/5005]	 Loss 3.770, Top 1-error 47.863, Top 5-error 23.663
Train with distillation: [Epoch 24/100][Batch 3000/5005]	 Loss 3.774, Top 1-error 47.917, Top 5-error 23.686
Train with distillation: [Epoch 24/100][Batch 3500/5005]	 Loss 3.778, Top 1-error 47.945, Top 5-error 23.725
Train with distillation: [Epoch 24/100][Batch 4000/5005]	 Loss 3.781, Top 1-error 47.990, Top 5-error 23.747
Train with distillation: [Epoch 24/100][Batch 4500/5005]	 Loss 3.783, Top 1-error 48.021, Top 5-error 23.789
Train with distillation: [Epoch 24/100][Batch 5000/5005]	 Loss 3.784, Top 1-error 48.060, Top 5-error 23.802
Train 	 Time Taken: 2937.54 sec
Test (on val set): [Epoch 24/100][Batch 0/196]	Time 2.059 (2.059)	Loss 1.2648 (1.2648)	Top 1-err 39.4531 (39.4531)	Top 5-err 11.7188 (11.7188)
* Epoch: [24/100]	 Top 1-err 46.958  Top 5-err 21.012	 Test Loss 1.981
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 25/100][Batch 0/5005]	 Loss 4.030, Top 1-error 51.953, Top 5-error 23.828
Train with distillation: [Epoch 25/100][Batch 500/5005]	 Loss 3.743, Top 1-error 47.641, Top 5-error 23.399
Train with distillation: [Epoch 25/100][Batch 1000/5005]	 Loss 3.749, Top 1-error 47.734, Top 5-error 23.478
Train with distillation: [Epoch 25/100][Batch 1500/5005]	 Loss 3.755, Top 1-error 47.779, Top 5-error 23.508
Train with distillation: [Epoch 25/100][Batch 2000/5005]	 Loss 3.759, Top 1-error 47.808, Top 5-error 23.566
Train with distillation: [Epoch 25/100][Batch 2500/5005]	 Loss 3.764, Top 1-error 47.862, Top 5-error 23.638
Train with distillation: [Epoch 25/100][Batch 3000/5005]	 Loss 3.769, Top 1-error 47.948, Top 5-error 23.688
Train with distillation: [Epoch 25/100][Batch 3500/5005]	 Loss 3.770, Top 1-error 47.969, Top 5-error 23.687
Train with distillation: [Epoch 25/100][Batch 4000/5005]	 Loss 3.772, Top 1-error 47.988, Top 5-error 23.695
Train with distillation: [Epoch 25/100][Batch 4500/5005]	 Loss 3.775, Top 1-error 48.025, Top 5-error 23.735
Train with distillation: [Epoch 25/100][Batch 5000/5005]	 Loss 3.775, Top 1-error 48.039, Top 5-error 23.726
Train 	 Time Taken: 2943.03 sec
Test (on val set): [Epoch 25/100][Batch 0/196]	Time 1.759 (1.759)	Loss 1.2319 (1.2319)	Top 1-err 34.7656 (34.7656)	Top 5-err 9.3750 (9.3750)
* Epoch: [25/100]	 Top 1-err 45.870  Top 5-err 20.114	 Test Loss 1.934
Current best accuracy (top-1 and 5 error): 45.416 19.9
Train with distillation: [Epoch 26/100][Batch 0/5005]	 Loss 3.771, Top 1-error 46.484, Top 5-error 21.875
Train with distillation: [Epoch 26/100][Batch 500/5005]	 Loss 3.733, Top 1-error 47.557, Top 5-error 23.303
Train with distillation: [Epoch 26/100][Batch 1000/5005]	 Loss 3.742, Top 1-error 47.627, Top 5-error 23.379
Train with distillation: [Epoch 26/100][Batch 1500/5005]	 Loss 3.751, Top 1-error 47.686, Top 5-error 23.426
Train with distillation: [Epoch 26/100][Batch 2000/5005]	 Loss 3.757, Top 1-error 47.802, Top 5-error 23.518
Train with distillation: [Epoch 26/100][Batch 2500/5005]	 Loss 3.761, Top 1-error 47.893, Top 5-error 23.552
Train with distillation: [Epoch 26/100][Batch 3000/5005]	 Loss 3.763, Top 1-error 47.918, Top 5-error 23.567
Train with distillation: [Epoch 26/100][Batch 3500/5005]	 Loss 3.765, Top 1-error 47.976, Top 5-error 23.602
Train with distillation: [Epoch 26/100][Batch 4000/5005]	 Loss 3.768, Top 1-error 48.010, Top 5-error 23.613
Train with distillation: [Epoch 26/100][Batch 4500/5005]	 Loss 3.770, Top 1-error 48.013, Top 5-error 23.633
Train with distillation: [Epoch 26/100][Batch 5000/5005]	 Loss 3.770, Top 1-error 48.035, Top 5-error 23.657
Train 	 Time Taken: 2943.38 sec
Test (on val set): [Epoch 26/100][Batch 0/196]	Time 1.982 (1.982)	Loss 0.9653 (0.9653)	Top 1-err 28.5156 (28.5156)	Top 5-err 5.4688 (5.4688)
* Epoch: [26/100]	 Top 1-err 45.150  Top 5-err 20.080	 Test Loss 1.906
Current best accuracy (top-1 and 5 error): 45.15 20.08
Train with distillation: [Epoch 27/100][Batch 0/5005]	 Loss 3.539, Top 1-error 40.234, Top 5-error 20.312
Train with distillation: [Epoch 27/100][Batch 500/5005]	 Loss 3.714, Top 1-error 47.323, Top 5-error 23.196
Train with distillation: [Epoch 27/100][Batch 1000/5005]	 Loss 3.725, Top 1-error 47.525, Top 5-error 23.279
Train with distillation: [Epoch 27/100][Batch 1500/5005]	 Loss 3.734, Top 1-error 47.596, Top 5-error 23.368
Train with distillation: [Epoch 27/100][Batch 2000/5005]	 Loss 3.739, Top 1-error 47.677, Top 5-error 23.419
Train with distillation: [Epoch 27/100][Batch 2500/5005]	 Loss 3.745, Top 1-error 47.760, Top 5-error 23.474
Train with distillation: [Epoch 27/100][Batch 3000/5005]	 Loss 3.750, Top 1-error 47.778, Top 5-error 23.525
Train with distillation: [Epoch 27/100][Batch 3500/5005]	 Loss 3.752, Top 1-error 47.811, Top 5-error 23.535
Train with distillation: [Epoch 27/100][Batch 4000/5005]	 Loss 3.755, Top 1-error 47.836, Top 5-error 23.583
Train with distillation: [Epoch 27/100][Batch 4500/5005]	 Loss 3.759, Top 1-error 47.907, Top 5-error 23.628
Train with distillation: [Epoch 27/100][Batch 5000/5005]	 Loss 3.761, Top 1-error 47.930, Top 5-error 23.647
Train 	 Time Taken: 2942.58 sec
Test (on val set): [Epoch 27/100][Batch 0/196]	Time 1.931 (1.931)	Loss 1.4408 (1.4408)	Top 1-err 41.7969 (41.7969)	Top 5-err 12.1094 (12.1094)
* Epoch: [27/100]	 Top 1-err 45.958  Top 5-err 20.588	 Test Loss 1.952
Current best accuracy (top-1 and 5 error): 45.15 20.08
Train with distillation: [Epoch 28/100][Batch 0/5005]	 Loss 3.647, Top 1-error 46.094, Top 5-error 21.875
Train with distillation: [Epoch 28/100][Batch 500/5005]	 Loss 3.721, Top 1-error 47.506, Top 5-error 23.087
Train with distillation: [Epoch 28/100][Batch 1000/5005]	 Loss 3.734, Top 1-error 47.667, Top 5-error 23.282
Train with distillation: [Epoch 28/100][Batch 1500/5005]	 Loss 3.740, Top 1-error 47.738, Top 5-error 23.360
Train with distillation: [Epoch 28/100][Batch 2000/5005]	 Loss 3.743, Top 1-error 47.796, Top 5-error 23.395
Train with distillation: [Epoch 28/100][Batch 2500/5005]	 Loss 3.744, Top 1-error 47.826, Top 5-error 23.427
Train with distillation: [Epoch 28/100][Batch 3000/5005]	 Loss 3.746, Top 1-error 47.851, Top 5-error 23.466
Train with distillation: [Epoch 28/100][Batch 3500/5005]	 Loss 3.750, Top 1-error 47.889, Top 5-error 23.495
Train with distillation: [Epoch 28/100][Batch 4000/5005]	 Loss 3.752, Top 1-error 47.892, Top 5-error 23.547
Train with distillation: [Epoch 28/100][Batch 4500/5005]	 Loss 3.755, Top 1-error 47.910, Top 5-error 23.574
Train with distillation: [Epoch 28/100][Batch 5000/5005]	 Loss 3.756, Top 1-error 47.941, Top 5-error 23.587
Train 	 Time Taken: 2930.29 sec
Test (on val set): [Epoch 28/100][Batch 0/196]	Time 1.955 (1.955)	Loss 1.4257 (1.4257)	Top 1-err 37.1094 (37.1094)	Top 5-err 15.2344 (15.2344)
* Epoch: [28/100]	 Top 1-err 45.756  Top 5-err 20.416	 Test Loss 1.935
Current best accuracy (top-1 and 5 error): 45.15 20.08
Train with distillation: [Epoch 29/100][Batch 0/5005]	 Loss 4.128, Top 1-error 53.906, Top 5-error 26.953
Train with distillation: [Epoch 29/100][Batch 500/5005]	 Loss 3.720, Top 1-error 47.462, Top 5-error 23.331
Train with distillation: [Epoch 29/100][Batch 1000/5005]	 Loss 3.725, Top 1-error 47.451, Top 5-error 23.300
Train with distillation: [Epoch 29/100][Batch 1500/5005]	 Loss 3.729, Top 1-error 47.480, Top 5-error 23.332
Train with distillation: [Epoch 29/100][Batch 2000/5005]	 Loss 3.736, Top 1-error 47.564, Top 5-error 23.409
Train with distillation: [Epoch 29/100][Batch 2500/5005]	 Loss 3.741, Top 1-error 47.646, Top 5-error 23.468
Train with distillation: [Epoch 29/100][Batch 3000/5005]	 Loss 3.743, Top 1-error 47.697, Top 5-error 23.498
Train with distillation: [Epoch 29/100][Batch 3500/5005]	 Loss 3.745, Top 1-error 47.740, Top 5-error 23.510
Train with distillation: [Epoch 29/100][Batch 4000/5005]	 Loss 3.748, Top 1-error 47.767, Top 5-error 23.526
Train with distillation: [Epoch 29/100][Batch 4500/5005]	 Loss 3.751, Top 1-error 47.803, Top 5-error 23.561
Train with distillation: [Epoch 29/100][Batch 5000/5005]	 Loss 3.753, Top 1-error 47.825, Top 5-error 23.573
Train 	 Time Taken: 2899.45 sec
Test (on val set): [Epoch 29/100][Batch 0/196]	Time 1.794 (1.794)	Loss 1.2083 (1.2083)	Top 1-err 30.0781 (30.0781)	Top 5-err 9.7656 (9.7656)
* Epoch: [29/100]	 Top 1-err 45.762  Top 5-err 20.488	 Test Loss 1.943
Current best accuracy (top-1 and 5 error): 45.15 20.08
Train with distillation: [Epoch 30/100][Batch 0/5005]	 Loss 3.558, Top 1-error 43.750, Top 5-error 23.438
Train with distillation: [Epoch 30/100][Batch 500/5005]	 Loss 3.170, Top 1-error 42.261, Top 5-error 19.750
Train with distillation: [Epoch 30/100][Batch 1000/5005]	 Loss 3.086, Top 1-error 41.556, Top 5-error 19.241
Train with distillation: [Epoch 30/100][Batch 1500/5005]	 Loss 3.032, Top 1-error 41.103, Top 5-error 18.881
Train with distillation: [Epoch 30/100][Batch 2000/5005]	 Loss 2.998, Top 1-error 40.803, Top 5-error 18.677
Train with distillation: [Epoch 30/100][Batch 2500/5005]	 Loss 2.968, Top 1-error 40.491, Top 5-error 18.488
Train with distillation: [Epoch 30/100][Batch 3000/5005]	 Loss 2.944, Top 1-error 40.269, Top 5-error 18.346
Train with distillation: [Epoch 30/100][Batch 3500/5005]	 Loss 2.926, Top 1-error 40.106, Top 5-error 18.238
Train with distillation: [Epoch 30/100][Batch 4000/5005]	 Loss 2.908, Top 1-error 39.918, Top 5-error 18.118
Train with distillation: [Epoch 30/100][Batch 4500/5005]	 Loss 2.894, Top 1-error 39.819, Top 5-error 18.028
Train with distillation: [Epoch 30/100][Batch 5000/5005]	 Loss 2.880, Top 1-error 39.687, Top 5-error 17.938
Train 	 Time Taken: 2878.07 sec
Test (on val set): [Epoch 30/100][Batch 0/196]	Time 1.890 (1.890)	Loss 0.8405 (0.8405)	Top 1-err 23.4375 (23.4375)	Top 5-err 4.2969 (4.2969)
* Epoch: [30/100]	 Top 1-err 34.470  Top 5-err 12.626	 Test Loss 1.380
Current best accuracy (top-1 and 5 error): 34.47 12.626
Train with distillation: [Epoch 31/100][Batch 0/5005]	 Loss 2.715, Top 1-error 39.844, Top 5-error 15.234
Train with distillation: [Epoch 31/100][Batch 500/5005]	 Loss 2.725, Top 1-error 38.142, Top 5-error 16.783
Train with distillation: [Epoch 31/100][Batch 1000/5005]	 Loss 2.721, Top 1-error 38.138, Top 5-error 16.764
Train with distillation: [Epoch 31/100][Batch 1500/5005]	 Loss 2.718, Top 1-error 38.046, Top 5-error 16.748
Train with distillation: [Epoch 31/100][Batch 2000/5005]	 Loss 2.718, Top 1-error 38.042, Top 5-error 16.800
Train with distillation: [Epoch 31/100][Batch 2500/5005]	 Loss 2.714, Top 1-error 37.989, Top 5-error 16.773
Train with distillation: [Epoch 31/100][Batch 3000/5005]	 Loss 2.709, Top 1-error 37.957, Top 5-error 16.729
Train with distillation: [Epoch 31/100][Batch 3500/5005]	 Loss 2.707, Top 1-error 37.916, Top 5-error 16.717
Train with distillation: [Epoch 31/100][Batch 4000/5005]	 Loss 2.703, Top 1-error 37.884, Top 5-error 16.718
Train with distillation: [Epoch 31/100][Batch 4500/5005]	 Loss 2.701, Top 1-error 37.878, Top 5-error 16.712
Train with distillation: [Epoch 31/100][Batch 5000/5005]	 Loss 2.697, Top 1-error 37.846, Top 5-error 16.700
Train 	 Time Taken: 2845.25 sec
Test (on val set): [Epoch 31/100][Batch 0/196]	Time 1.808 (1.808)	Loss 0.7934 (0.7934)	Top 1-err 23.8281 (23.8281)	Top 5-err 4.6875 (4.6875)
* Epoch: [31/100]	 Top 1-err 33.564  Top 5-err 12.010	 Test Loss 1.343
Current best accuracy (top-1 and 5 error): 33.564 12.01
Train with distillation: [Epoch 32/100][Batch 0/5005]	 Loss 2.568, Top 1-error 38.672, Top 5-error 14.844
Train with distillation: [Epoch 32/100][Batch 500/5005]	 Loss 2.642, Top 1-error 37.391, Top 5-error 16.273
Train with distillation: [Epoch 32/100][Batch 1000/5005]	 Loss 2.641, Top 1-error 37.341, Top 5-error 16.245
Train with distillation: [Epoch 32/100][Batch 1500/5005]	 Loss 2.639, Top 1-error 37.286, Top 5-error 16.252
Train with distillation: [Epoch 32/100][Batch 2000/5005]	 Loss 2.638, Top 1-error 37.262, Top 5-error 16.246
Train with distillation: [Epoch 32/100][Batch 2500/5005]	 Loss 2.637, Top 1-error 37.241, Top 5-error 16.262
Train with distillation: [Epoch 32/100][Batch 3000/5005]	 Loss 2.636, Top 1-error 37.276, Top 5-error 16.273
Train with distillation: [Epoch 32/100][Batch 3500/5005]	 Loss 2.635, Top 1-error 37.262, Top 5-error 16.277
Train with distillation: [Epoch 32/100][Batch 4000/5005]	 Loss 2.635, Top 1-error 37.266, Top 5-error 16.279
Train with distillation: [Epoch 32/100][Batch 4500/5005]	 Loss 2.634, Top 1-error 37.251, Top 5-error 16.265
Train with distillation: [Epoch 32/100][Batch 5000/5005]	 Loss 2.632, Top 1-error 37.244, Top 5-error 16.249
Train 	 Time Taken: 2825.51 sec
Test (on val set): [Epoch 32/100][Batch 0/196]	Time 1.989 (1.989)	Loss 0.8115 (0.8115)	Top 1-err 24.6094 (24.6094)	Top 5-err 3.9062 (3.9062)
* Epoch: [32/100]	 Top 1-err 33.182  Top 5-err 11.800	 Test Loss 1.329
Current best accuracy (top-1 and 5 error): 33.182 11.8
Train with distillation: [Epoch 33/100][Batch 0/5005]	 Loss 2.695, Top 1-error 39.453, Top 5-error 18.359
Train with distillation: [Epoch 33/100][Batch 500/5005]	 Loss 2.589, Top 1-error 36.673, Top 5-error 15.882
Train with distillation: [Epoch 33/100][Batch 1000/5005]	 Loss 2.593, Top 1-error 36.773, Top 5-error 15.963
Train with distillation: [Epoch 33/100][Batch 1500/5005]	 Loss 2.592, Top 1-error 36.668, Top 5-error 15.921
Train with distillation: [Epoch 33/100][Batch 2000/5005]	 Loss 2.594, Top 1-error 36.683, Top 5-error 15.904
Train with distillation: [Epoch 33/100][Batch 2500/5005]	 Loss 2.593, Top 1-error 36.681, Top 5-error 15.896
Train with distillation: [Epoch 33/100][Batch 3000/5005]	 Loss 2.593, Top 1-error 36.723, Top 5-error 15.920
Train with distillation: [Epoch 33/100][Batch 3500/5005]	 Loss 2.592, Top 1-error 36.723, Top 5-error 15.918
Train with distillation: [Epoch 33/100][Batch 4000/5005]	 Loss 2.593, Top 1-error 36.730, Top 5-error 15.925
Train with distillation: [Epoch 33/100][Batch 4500/5005]	 Loss 2.594, Top 1-error 36.752, Top 5-error 15.948
Train with distillation: [Epoch 33/100][Batch 5000/5005]	 Loss 2.593, Top 1-error 36.741, Top 5-error 15.937
Train 	 Time Taken: 2790.22 sec
Test (on val set): [Epoch 33/100][Batch 0/196]	Time 1.813 (1.813)	Loss 0.7834 (0.7834)	Top 1-err 23.0469 (23.0469)	Top 5-err 4.2969 (4.2969)
* Epoch: [33/100]	 Top 1-err 32.824  Top 5-err 11.584	 Test Loss 1.315
Current best accuracy (top-1 and 5 error): 32.824 11.584
Train with distillation: [Epoch 34/100][Batch 0/5005]	 Loss 2.552, Top 1-error 38.672, Top 5-error 17.578
Train with distillation: [Epoch 34/100][Batch 500/5005]	 Loss 2.559, Top 1-error 36.348, Top 5-error 15.624
Train with distillation: [Epoch 34/100][Batch 1000/5005]	 Loss 2.561, Top 1-error 36.376, Top 5-error 15.630
Train with distillation: [Epoch 34/100][Batch 1500/5005]	 Loss 2.560, Top 1-error 36.306, Top 5-error 15.625
Train with distillation: [Epoch 34/100][Batch 2000/5005]	 Loss 2.561, Top 1-error 36.384, Top 5-error 15.633
Train with distillation: [Epoch 34/100][Batch 2500/5005]	 Loss 2.564, Top 1-error 36.453, Top 5-error 15.687
Train with distillation: [Epoch 34/100][Batch 3000/5005]	 Loss 2.564, Top 1-error 36.448, Top 5-error 15.671
Train with distillation: [Epoch 34/100][Batch 3500/5005]	 Loss 2.565, Top 1-error 36.465, Top 5-error 15.682
Train with distillation: [Epoch 34/100][Batch 4000/5005]	 Loss 2.566, Top 1-error 36.493, Top 5-error 15.698
Train with distillation: [Epoch 34/100][Batch 4500/5005]	 Loss 2.567, Top 1-error 36.502, Top 5-error 15.704
Train with distillation: [Epoch 34/100][Batch 5000/5005]	 Loss 2.567, Top 1-error 36.505, Top 5-error 15.712
Train 	 Time Taken: 2779.31 sec
Test (on val set): [Epoch 34/100][Batch 0/196]	Time 2.017 (2.017)	Loss 0.7189 (0.7189)	Top 1-err 21.4844 (21.4844)	Top 5-err 3.1250 (3.1250)
* Epoch: [34/100]	 Top 1-err 32.632  Top 5-err 11.442	 Test Loss 1.309
Current best accuracy (top-1 and 5 error): 32.632 11.442
Train with distillation: [Epoch 35/100][Batch 0/5005]	 Loss 2.615, Top 1-error 38.672, Top 5-error 16.797
Train with distillation: [Epoch 35/100][Batch 500/5005]	 Loss 2.545, Top 1-error 36.076, Top 5-error 15.421
Train with distillation: [Epoch 35/100][Batch 1000/5005]	 Loss 2.548, Top 1-error 36.200, Top 5-error 15.472
Train with distillation: [Epoch 35/100][Batch 1500/5005]	 Loss 2.551, Top 1-error 36.225, Top 5-error 15.526
Train with distillation: [Epoch 35/100][Batch 2000/5005]	 Loss 2.551, Top 1-error 36.190, Top 5-error 15.551
Train with distillation: [Epoch 35/100][Batch 2500/5005]	 Loss 2.551, Top 1-error 36.159, Top 5-error 15.558
Train with distillation: [Epoch 35/100][Batch 3000/5005]	 Loss 2.551, Top 1-error 36.163, Top 5-error 15.546
Train with distillation: [Epoch 35/100][Batch 3500/5005]	 Loss 2.552, Top 1-error 36.188, Top 5-error 15.571
Train with distillation: [Epoch 35/100][Batch 4000/5005]	 Loss 2.552, Top 1-error 36.205, Top 5-error 15.568
Train with distillation: [Epoch 35/100][Batch 4500/5005]	 Loss 2.552, Top 1-error 36.216, Top 5-error 15.591
Train with distillation: [Epoch 35/100][Batch 5000/5005]	 Loss 2.553, Top 1-error 36.224, Top 5-error 15.591
Train 	 Time Taken: 2769.03 sec
Test (on val set): [Epoch 35/100][Batch 0/196]	Time 1.843 (1.843)	Loss 0.7712 (0.7712)	Top 1-err 23.0469 (23.0469)	Top 5-err 2.7344 (2.7344)
* Epoch: [35/100]	 Top 1-err 32.546  Top 5-err 11.424	 Test Loss 1.304
Current best accuracy (top-1 and 5 error): 32.546 11.424
Train with distillation: [Epoch 36/100][Batch 0/5005]	 Loss 2.461, Top 1-error 36.328, Top 5-error 16.406
Train with distillation: [Epoch 36/100][Batch 500/5005]	 Loss 2.532, Top 1-error 35.810, Top 5-error 15.385
Train with distillation: [Epoch 36/100][Batch 1000/5005]	 Loss 2.532, Top 1-error 35.864, Top 5-error 15.412
Train with distillation: [Epoch 36/100][Batch 1500/5005]	 Loss 2.532, Top 1-error 35.901, Top 5-error 15.391
Train with distillation: [Epoch 36/100][Batch 2000/5005]	 Loss 2.535, Top 1-error 35.949, Top 5-error 15.410
Train with distillation: [Epoch 36/100][Batch 2500/5005]	 Loss 2.536, Top 1-error 35.993, Top 5-error 15.411
Train with distillation: [Epoch 36/100][Batch 3000/5005]	 Loss 2.537, Top 1-error 36.009, Top 5-error 15.428
Train with distillation: [Epoch 36/100][Batch 3500/5005]	 Loss 2.539, Top 1-error 36.036, Top 5-error 15.450
Train with distillation: [Epoch 36/100][Batch 4000/5005]	 Loss 2.539, Top 1-error 36.052, Top 5-error 15.452
Train with distillation: [Epoch 36/100][Batch 4500/5005]	 Loss 2.539, Top 1-error 36.060, Top 5-error 15.451
Train with distillation: [Epoch 36/100][Batch 5000/5005]	 Loss 2.540, Top 1-error 36.079, Top 5-error 15.467
Train 	 Time Taken: 2752.90 sec
Test (on val set): [Epoch 36/100][Batch 0/196]	Time 2.063 (2.063)	Loss 0.7914 (0.7914)	Top 1-err 26.1719 (26.1719)	Top 5-err 3.1250 (3.1250)
* Epoch: [36/100]	 Top 1-err 32.306  Top 5-err 11.460	 Test Loss 1.295
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 37/100][Batch 0/5005]	 Loss 2.567, Top 1-error 36.328, Top 5-error 16.406
Train with distillation: [Epoch 37/100][Batch 500/5005]	 Loss 2.523, Top 1-error 35.811, Top 5-error 15.344
Train with distillation: [Epoch 37/100][Batch 1000/5005]	 Loss 2.524, Top 1-error 35.826, Top 5-error 15.339
Train with distillation: [Epoch 37/100][Batch 1500/5005]	 Loss 2.528, Top 1-error 35.878, Top 5-error 15.402
Train with distillation: [Epoch 37/100][Batch 2000/5005]	 Loss 2.528, Top 1-error 35.862, Top 5-error 15.400
Train with distillation: [Epoch 37/100][Batch 2500/5005]	 Loss 2.527, Top 1-error 35.849, Top 5-error 15.371
Train with distillation: [Epoch 37/100][Batch 3000/5005]	 Loss 2.528, Top 1-error 35.835, Top 5-error 15.370
Train with distillation: [Epoch 37/100][Batch 3500/5005]	 Loss 2.531, Top 1-error 35.881, Top 5-error 15.417
Train with distillation: [Epoch 37/100][Batch 4000/5005]	 Loss 2.532, Top 1-error 35.893, Top 5-error 15.417
Train with distillation: [Epoch 37/100][Batch 4500/5005]	 Loss 2.533, Top 1-error 35.932, Top 5-error 15.431
Train with distillation: [Epoch 37/100][Batch 5000/5005]	 Loss 2.533, Top 1-error 35.937, Top 5-error 15.433
Train 	 Time Taken: 2720.90 sec
Test (on val set): [Epoch 37/100][Batch 0/196]	Time 1.858 (1.858)	Loss 0.7087 (0.7087)	Top 1-err 22.6562 (22.6562)	Top 5-err 3.1250 (3.1250)
* Epoch: [37/100]	 Top 1-err 33.076  Top 5-err 12.002	 Test Loss 1.334
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 38/100][Batch 0/5005]	 Loss 2.441, Top 1-error 32.422, Top 5-error 12.891
Train with distillation: [Epoch 38/100][Batch 500/5005]	 Loss 2.515, Top 1-error 35.418, Top 5-error 15.138
Train with distillation: [Epoch 38/100][Batch 1000/5005]	 Loss 2.517, Top 1-error 35.593, Top 5-error 15.236
Train with distillation: [Epoch 38/100][Batch 1500/5005]	 Loss 2.524, Top 1-error 35.686, Top 5-error 15.280
Train with distillation: [Epoch 38/100][Batch 2000/5005]	 Loss 2.525, Top 1-error 35.727, Top 5-error 15.303
Train with distillation: [Epoch 38/100][Batch 2500/5005]	 Loss 2.528, Top 1-error 35.819, Top 5-error 15.340
Train with distillation: [Epoch 38/100][Batch 3000/5005]	 Loss 2.531, Top 1-error 35.887, Top 5-error 15.374
Train with distillation: [Epoch 38/100][Batch 3500/5005]	 Loss 2.532, Top 1-error 35.895, Top 5-error 15.391
Train with distillation: [Epoch 38/100][Batch 4000/5005]	 Loss 2.532, Top 1-error 35.896, Top 5-error 15.383
Train with distillation: [Epoch 38/100][Batch 4500/5005]	 Loss 2.533, Top 1-error 35.922, Top 5-error 15.407
Train with distillation: [Epoch 38/100][Batch 5000/5005]	 Loss 2.535, Top 1-error 35.935, Top 5-error 15.429
Train 	 Time Taken: 2706.32 sec
Test (on val set): [Epoch 38/100][Batch 0/196]	Time 1.999 (1.999)	Loss 0.7161 (0.7161)	Top 1-err 21.8750 (21.8750)	Top 5-err 3.5156 (3.5156)
* Epoch: [38/100]	 Top 1-err 32.374  Top 5-err 11.408	 Test Loss 1.297
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 39/100][Batch 0/5005]	 Loss 2.777, Top 1-error 37.500, Top 5-error 18.359
Train with distillation: [Epoch 39/100][Batch 500/5005]	 Loss 2.513, Top 1-error 35.526, Top 5-error 15.201
Train with distillation: [Epoch 39/100][Batch 1000/5005]	 Loss 2.517, Top 1-error 35.612, Top 5-error 15.257
Train with distillation: [Epoch 39/100][Batch 1500/5005]	 Loss 2.520, Top 1-error 35.708, Top 5-error 15.255
Train with distillation: [Epoch 39/100][Batch 2000/5005]	 Loss 2.524, Top 1-error 35.744, Top 5-error 15.310
Train with distillation: [Epoch 39/100][Batch 2500/5005]	 Loss 2.526, Top 1-error 35.777, Top 5-error 15.319
Train with distillation: [Epoch 39/100][Batch 3000/5005]	 Loss 2.529, Top 1-error 35.829, Top 5-error 15.364
Train with distillation: [Epoch 39/100][Batch 3500/5005]	 Loss 2.532, Top 1-error 35.865, Top 5-error 15.392
Train with distillation: [Epoch 39/100][Batch 4000/5005]	 Loss 2.532, Top 1-error 35.869, Top 5-error 15.380
Train with distillation: [Epoch 39/100][Batch 4500/5005]	 Loss 2.533, Top 1-error 35.907, Top 5-error 15.387
Train with distillation: [Epoch 39/100][Batch 5000/5005]	 Loss 2.535, Top 1-error 35.933, Top 5-error 15.394
Train 	 Time Taken: 2690.71 sec
Test (on val set): [Epoch 39/100][Batch 0/196]	Time 1.764 (1.764)	Loss 0.6909 (0.6909)	Top 1-err 20.7031 (20.7031)	Top 5-err 2.7344 (2.7344)
* Epoch: [39/100]	 Top 1-err 32.634  Top 5-err 11.590	 Test Loss 1.310
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 40/100][Batch 0/5005]	 Loss 2.583, Top 1-error 32.422, Top 5-error 16.016
Train with distillation: [Epoch 40/100][Batch 500/5005]	 Loss 2.522, Top 1-error 35.591, Top 5-error 15.257
Train with distillation: [Epoch 40/100][Batch 1000/5005]	 Loss 2.528, Top 1-error 35.681, Top 5-error 15.336
Train with distillation: [Epoch 40/100][Batch 1500/5005]	 Loss 2.528, Top 1-error 35.764, Top 5-error 15.294
Train with distillation: [Epoch 40/100][Batch 2000/5005]	 Loss 2.532, Top 1-error 35.829, Top 5-error 15.316
Train with distillation: [Epoch 40/100][Batch 2500/5005]	 Loss 2.530, Top 1-error 35.774, Top 5-error 15.290
Train with distillation: [Epoch 40/100][Batch 3000/5005]	 Loss 2.530, Top 1-error 35.768, Top 5-error 15.286
Train with distillation: [Epoch 40/100][Batch 3500/5005]	 Loss 2.533, Top 1-error 35.791, Top 5-error 15.327
Train with distillation: [Epoch 40/100][Batch 4000/5005]	 Loss 2.534, Top 1-error 35.828, Top 5-error 15.324
Train with distillation: [Epoch 40/100][Batch 4500/5005]	 Loss 2.535, Top 1-error 35.853, Top 5-error 15.331
Train with distillation: [Epoch 40/100][Batch 5000/5005]	 Loss 2.536, Top 1-error 35.880, Top 5-error 15.344
Train 	 Time Taken: 2680.85 sec
Test (on val set): [Epoch 40/100][Batch 0/196]	Time 1.883 (1.883)	Loss 0.7950 (0.7950)	Top 1-err 23.0469 (23.0469)	Top 5-err 3.9062 (3.9062)
* Epoch: [40/100]	 Top 1-err 32.734  Top 5-err 11.498	 Test Loss 1.310
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 41/100][Batch 0/5005]	 Loss 2.516, Top 1-error 34.766, Top 5-error 14.062
Train with distillation: [Epoch 41/100][Batch 500/5005]	 Loss 2.529, Top 1-error 35.694, Top 5-error 15.383
Train with distillation: [Epoch 41/100][Batch 1000/5005]	 Loss 2.521, Top 1-error 35.622, Top 5-error 15.222
Train with distillation: [Epoch 41/100][Batch 1500/5005]	 Loss 2.527, Top 1-error 35.719, Top 5-error 15.282
Train with distillation: [Epoch 41/100][Batch 2000/5005]	 Loss 2.532, Top 1-error 35.809, Top 5-error 15.316
Train with distillation: [Epoch 41/100][Batch 2500/5005]	 Loss 2.533, Top 1-error 35.818, Top 5-error 15.317
Train with distillation: [Epoch 41/100][Batch 3000/5005]	 Loss 2.535, Top 1-error 35.858, Top 5-error 15.339
Train with distillation: [Epoch 41/100][Batch 3500/5005]	 Loss 2.536, Top 1-error 35.866, Top 5-error 15.327
Train with distillation: [Epoch 41/100][Batch 4000/5005]	 Loss 2.539, Top 1-error 35.914, Top 5-error 15.368
Train with distillation: [Epoch 41/100][Batch 4500/5005]	 Loss 2.540, Top 1-error 35.946, Top 5-error 15.381
Train with distillation: [Epoch 41/100][Batch 5000/5005]	 Loss 2.543, Top 1-error 35.997, Top 5-error 15.420
Train 	 Time Taken: 2672.88 sec
Test (on val set): [Epoch 41/100][Batch 0/196]	Time 1.797 (1.797)	Loss 0.6759 (0.6759)	Top 1-err 21.4844 (21.4844)	Top 5-err 3.9062 (3.9062)
* Epoch: [41/100]	 Top 1-err 32.774  Top 5-err 11.600	 Test Loss 1.311
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 42/100][Batch 0/5005]	 Loss 2.428, Top 1-error 33.594, Top 5-error 15.234
Train with distillation: [Epoch 42/100][Batch 500/5005]	 Loss 2.517, Top 1-error 35.455, Top 5-error 15.276
Train with distillation: [Epoch 42/100][Batch 1000/5005]	 Loss 2.523, Top 1-error 35.508, Top 5-error 15.262
Train with distillation: [Epoch 42/100][Batch 1500/5005]	 Loss 2.525, Top 1-error 35.535, Top 5-error 15.224
Train with distillation: [Epoch 42/100][Batch 2000/5005]	 Loss 2.529, Top 1-error 35.634, Top 5-error 15.256
Train with distillation: [Epoch 42/100][Batch 2500/5005]	 Loss 2.534, Top 1-error 35.746, Top 5-error 15.300
Train with distillation: [Epoch 42/100][Batch 3000/5005]	 Loss 2.538, Top 1-error 35.806, Top 5-error 15.339
Train with distillation: [Epoch 42/100][Batch 3500/5005]	 Loss 2.540, Top 1-error 35.837, Top 5-error 15.357
Train with distillation: [Epoch 42/100][Batch 4000/5005]	 Loss 2.542, Top 1-error 35.883, Top 5-error 15.375
Train with distillation: [Epoch 42/100][Batch 4500/5005]	 Loss 2.544, Top 1-error 35.912, Top 5-error 15.390
Train with distillation: [Epoch 42/100][Batch 5000/5005]	 Loss 2.545, Top 1-error 35.915, Top 5-error 15.403
Train 	 Time Taken: 2653.17 sec
Test (on val set): [Epoch 42/100][Batch 0/196]	Time 1.791 (1.791)	Loss 0.6994 (0.6994)	Top 1-err 22.2656 (22.2656)	Top 5-err 3.1250 (3.1250)
* Epoch: [42/100]	 Top 1-err 32.674  Top 5-err 11.688	 Test Loss 1.313
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 43/100][Batch 0/5005]	 Loss 2.530, Top 1-error 36.719, Top 5-error 14.453
Train with distillation: [Epoch 43/100][Batch 500/5005]	 Loss 2.519, Top 1-error 35.397, Top 5-error 15.099
Train with distillation: [Epoch 43/100][Batch 1000/5005]	 Loss 2.528, Top 1-error 35.571, Top 5-error 15.221
Train with distillation: [Epoch 43/100][Batch 1500/5005]	 Loss 2.530, Top 1-error 35.579, Top 5-error 15.229
Train with distillation: [Epoch 43/100][Batch 2000/5005]	 Loss 2.534, Top 1-error 35.611, Top 5-error 15.243
Train with distillation: [Epoch 43/100][Batch 2500/5005]	 Loss 2.538, Top 1-error 35.677, Top 5-error 15.287
Train with distillation: [Epoch 43/100][Batch 3000/5005]	 Loss 2.540, Top 1-error 35.747, Top 5-error 15.312
Train with distillation: [Epoch 43/100][Batch 3500/5005]	 Loss 2.542, Top 1-error 35.803, Top 5-error 15.330
Train with distillation: [Epoch 43/100][Batch 4000/5005]	 Loss 2.543, Top 1-error 35.846, Top 5-error 15.335
Train with distillation: [Epoch 43/100][Batch 4500/5005]	 Loss 2.546, Top 1-error 35.880, Top 5-error 15.363
Train with distillation: [Epoch 43/100][Batch 5000/5005]	 Loss 2.548, Top 1-error 35.915, Top 5-error 15.387
Train 	 Time Taken: 2651.00 sec
Test (on val set): [Epoch 43/100][Batch 0/196]	Time 1.955 (1.955)	Loss 0.7257 (0.7257)	Top 1-err 21.8750 (21.8750)	Top 5-err 3.1250 (3.1250)
* Epoch: [43/100]	 Top 1-err 32.742  Top 5-err 11.612	 Test Loss 1.313
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 44/100][Batch 0/5005]	 Loss 2.648, Top 1-error 39.844, Top 5-error 19.922
Train with distillation: [Epoch 44/100][Batch 500/5005]	 Loss 2.527, Top 1-error 35.539, Top 5-error 15.114
Train with distillation: [Epoch 44/100][Batch 1000/5005]	 Loss 2.534, Top 1-error 35.657, Top 5-error 15.245
Train with distillation: [Epoch 44/100][Batch 1500/5005]	 Loss 2.535, Top 1-error 35.667, Top 5-error 15.195
Train with distillation: [Epoch 44/100][Batch 2000/5005]	 Loss 2.537, Top 1-error 35.703, Top 5-error 15.241
Train with distillation: [Epoch 44/100][Batch 2500/5005]	 Loss 2.543, Top 1-error 35.796, Top 5-error 15.306
Train with distillation: [Epoch 44/100][Batch 3000/5005]	 Loss 2.544, Top 1-error 35.817, Top 5-error 15.310
Train with distillation: [Epoch 44/100][Batch 3500/5005]	 Loss 2.547, Top 1-error 35.871, Top 5-error 15.336
Train with distillation: [Epoch 44/100][Batch 4000/5005]	 Loss 2.549, Top 1-error 35.898, Top 5-error 15.365
Train with distillation: [Epoch 44/100][Batch 4500/5005]	 Loss 2.551, Top 1-error 35.913, Top 5-error 15.389
Train with distillation: [Epoch 44/100][Batch 5000/5005]	 Loss 2.554, Top 1-error 35.962, Top 5-error 15.415
Train 	 Time Taken: 2639.36 sec
Test (on val set): [Epoch 44/100][Batch 0/196]	Time 1.889 (1.889)	Loss 0.7128 (0.7128)	Top 1-err 23.0469 (23.0469)	Top 5-err 3.9062 (3.9062)
* Epoch: [44/100]	 Top 1-err 32.614  Top 5-err 11.708	 Test Loss 1.316
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 45/100][Batch 0/5005]	 Loss 2.384, Top 1-error 31.641, Top 5-error 12.891
Train with distillation: [Epoch 45/100][Batch 500/5005]	 Loss 2.521, Top 1-error 35.470, Top 5-error 15.060
Train with distillation: [Epoch 45/100][Batch 1000/5005]	 Loss 2.526, Top 1-error 35.574, Top 5-error 14.994
Train with distillation: [Epoch 45/100][Batch 1500/5005]	 Loss 2.534, Top 1-error 35.732, Top 5-error 15.120
Train with distillation: [Epoch 45/100][Batch 2000/5005]	 Loss 2.542, Top 1-error 35.837, Top 5-error 15.215
Train with distillation: [Epoch 45/100][Batch 2500/5005]	 Loss 2.545, Top 1-error 35.862, Top 5-error 15.251
Train with distillation: [Epoch 45/100][Batch 3000/5005]	 Loss 2.549, Top 1-error 35.918, Top 5-error 15.306
Train with distillation: [Epoch 45/100][Batch 3500/5005]	 Loss 2.552, Top 1-error 35.948, Top 5-error 15.331
Train with distillation: [Epoch 45/100][Batch 4000/5005]	 Loss 2.553, Top 1-error 35.967, Top 5-error 15.352
Train with distillation: [Epoch 45/100][Batch 4500/5005]	 Loss 2.556, Top 1-error 35.997, Top 5-error 15.385
Train with distillation: [Epoch 45/100][Batch 5000/5005]	 Loss 2.557, Top 1-error 36.022, Top 5-error 15.387
Train 	 Time Taken: 2631.90 sec
Test (on val set): [Epoch 45/100][Batch 0/196]	Time 1.941 (1.941)	Loss 0.6403 (0.6403)	Top 1-err 19.9219 (19.9219)	Top 5-err 1.9531 (1.9531)
* Epoch: [45/100]	 Top 1-err 32.808  Top 5-err 11.718	 Test Loss 1.315
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 46/100][Batch 0/5005]	 Loss 2.570, Top 1-error 37.500, Top 5-error 14.844
Train with distillation: [Epoch 46/100][Batch 500/5005]	 Loss 2.543, Top 1-error 35.667, Top 5-error 15.283
Train with distillation: [Epoch 46/100][Batch 1000/5005]	 Loss 2.546, Top 1-error 35.759, Top 5-error 15.333
Train with distillation: [Epoch 46/100][Batch 1500/5005]	 Loss 2.550, Top 1-error 35.825, Top 5-error 15.339
Train with distillation: [Epoch 46/100][Batch 2000/5005]	 Loss 2.550, Top 1-error 35.851, Top 5-error 15.330
Train with distillation: [Epoch 46/100][Batch 2500/5005]	 Loss 2.552, Top 1-error 35.891, Top 5-error 15.322
Train with distillation: [Epoch 46/100][Batch 3000/5005]	 Loss 2.553, Top 1-error 35.925, Top 5-error 15.342
Train with distillation: [Epoch 46/100][Batch 3500/5005]	 Loss 2.555, Top 1-error 35.962, Top 5-error 15.365
Train with distillation: [Epoch 46/100][Batch 4000/5005]	 Loss 2.559, Top 1-error 36.020, Top 5-error 15.417
Train with distillation: [Epoch 46/100][Batch 4500/5005]	 Loss 2.560, Top 1-error 36.020, Top 5-error 15.422
Train with distillation: [Epoch 46/100][Batch 5000/5005]	 Loss 2.561, Top 1-error 36.032, Top 5-error 15.422
Train 	 Time Taken: 2614.67 sec
Test (on val set): [Epoch 46/100][Batch 0/196]	Time 1.768 (1.768)	Loss 0.6974 (0.6974)	Top 1-err 20.7031 (20.7031)	Top 5-err 3.9062 (3.9062)
* Epoch: [46/100]	 Top 1-err 32.662  Top 5-err 11.708	 Test Loss 1.314
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 47/100][Batch 0/5005]	 Loss 2.615, Top 1-error 36.328, Top 5-error 15.234
Train with distillation: [Epoch 47/100][Batch 500/5005]	 Loss 2.548, Top 1-error 35.788, Top 5-error 15.329
Train with distillation: [Epoch 47/100][Batch 1000/5005]	 Loss 2.549, Top 1-error 35.717, Top 5-error 15.318
Train with distillation: [Epoch 47/100][Batch 1500/5005]	 Loss 2.553, Top 1-error 35.863, Top 5-error 15.331
Train with distillation: [Epoch 47/100][Batch 2000/5005]	 Loss 2.555, Top 1-error 35.887, Top 5-error 15.334
Train with distillation: [Epoch 47/100][Batch 2500/5005]	 Loss 2.558, Top 1-error 35.887, Top 5-error 15.349
Train with distillation: [Epoch 47/100][Batch 3000/5005]	 Loss 2.560, Top 1-error 35.910, Top 5-error 15.365
Train with distillation: [Epoch 47/100][Batch 3500/5005]	 Loss 2.561, Top 1-error 35.927, Top 5-error 15.375
Train with distillation: [Epoch 47/100][Batch 4000/5005]	 Loss 2.562, Top 1-error 35.960, Top 5-error 15.399
Train with distillation: [Epoch 47/100][Batch 4500/5005]	 Loss 2.564, Top 1-error 35.978, Top 5-error 15.415
Train with distillation: [Epoch 47/100][Batch 5000/5005]	 Loss 2.566, Top 1-error 36.011, Top 5-error 15.447
Train 	 Time Taken: 2611.37 sec
Test (on val set): [Epoch 47/100][Batch 0/196]	Time 2.000 (2.000)	Loss 0.7724 (0.7724)	Top 1-err 23.0469 (23.0469)	Top 5-err 4.2969 (4.2969)
* Epoch: [47/100]	 Top 1-err 32.930  Top 5-err 11.876	 Test Loss 1.327
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 48/100][Batch 0/5005]	 Loss 2.506, Top 1-error 34.766, Top 5-error 13.672
Train with distillation: [Epoch 48/100][Batch 500/5005]	 Loss 2.548, Top 1-error 35.903, Top 5-error 15.358
Train with distillation: [Epoch 48/100][Batch 1000/5005]	 Loss 2.553, Top 1-error 35.877, Top 5-error 15.357
Train with distillation: [Epoch 48/100][Batch 1500/5005]	 Loss 2.553, Top 1-error 35.856, Top 5-error 15.342
Train with distillation: [Epoch 48/100][Batch 2000/5005]	 Loss 2.555, Top 1-error 35.914, Top 5-error 15.372
Train with distillation: [Epoch 48/100][Batch 2500/5005]	 Loss 2.559, Top 1-error 35.951, Top 5-error 15.410
Train with distillation: [Epoch 48/100][Batch 3000/5005]	 Loss 2.563, Top 1-error 36.015, Top 5-error 15.449
Train with distillation: [Epoch 48/100][Batch 3500/5005]	 Loss 2.565, Top 1-error 36.070, Top 5-error 15.489
Train with distillation: [Epoch 48/100][Batch 4000/5005]	 Loss 2.566, Top 1-error 36.070, Top 5-error 15.483
Train with distillation: [Epoch 48/100][Batch 4500/5005]	 Loss 2.566, Top 1-error 36.055, Top 5-error 15.477
Train with distillation: [Epoch 48/100][Batch 5000/5005]	 Loss 2.567, Top 1-error 36.082, Top 5-error 15.493
Train 	 Time Taken: 2592.75 sec
Test (on val set): [Epoch 48/100][Batch 0/196]	Time 1.757 (1.757)	Loss 0.7548 (0.7548)	Top 1-err 22.2656 (22.2656)	Top 5-err 3.5156 (3.5156)
* Epoch: [48/100]	 Top 1-err 33.036  Top 5-err 11.840	 Test Loss 1.323
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 49/100][Batch 0/5005]	 Loss 2.223, Top 1-error 29.297, Top 5-error 9.375
Train with distillation: [Epoch 49/100][Batch 500/5005]	 Loss 2.546, Top 1-error 35.709, Top 5-error 15.338
Train with distillation: [Epoch 49/100][Batch 1000/5005]	 Loss 2.551, Top 1-error 35.777, Top 5-error 15.326
Train with distillation: [Epoch 49/100][Batch 1500/5005]	 Loss 2.554, Top 1-error 35.781, Top 5-error 15.316
Train with distillation: [Epoch 49/100][Batch 2000/5005]	 Loss 2.559, Top 1-error 35.849, Top 5-error 15.395
Train with distillation: [Epoch 49/100][Batch 2500/5005]	 Loss 2.560, Top 1-error 35.867, Top 5-error 15.396
Train with distillation: [Epoch 49/100][Batch 3000/5005]	 Loss 2.564, Top 1-error 35.935, Top 5-error 15.434
Train with distillation: [Epoch 49/100][Batch 3500/5005]	 Loss 2.565, Top 1-error 35.964, Top 5-error 15.433
Train with distillation: [Epoch 49/100][Batch 4000/5005]	 Loss 2.567, Top 1-error 36.006, Top 5-error 15.459
Train with distillation: [Epoch 49/100][Batch 4500/5005]	 Loss 2.570, Top 1-error 36.045, Top 5-error 15.489
Train with distillation: [Epoch 49/100][Batch 5000/5005]	 Loss 2.571, Top 1-error 36.068, Top 5-error 15.488
Train 	 Time Taken: 2581.64 sec
Test (on val set): [Epoch 49/100][Batch 0/196]	Time 1.952 (1.952)	Loss 0.7343 (0.7343)	Top 1-err 23.0469 (23.0469)	Top 5-err 2.3438 (2.3438)
* Epoch: [49/100]	 Top 1-err 32.998  Top 5-err 11.876	 Test Loss 1.325
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 50/100][Batch 0/5005]	 Loss 2.469, Top 1-error 32.422, Top 5-error 12.500
Train with distillation: [Epoch 50/100][Batch 500/5005]	 Loss 2.549, Top 1-error 35.746, Top 5-error 15.112
Train with distillation: [Epoch 50/100][Batch 1000/5005]	 Loss 2.552, Top 1-error 35.781, Top 5-error 15.231
Train with distillation: [Epoch 50/100][Batch 1500/5005]	 Loss 2.556, Top 1-error 35.900, Top 5-error 15.274
Train with distillation: [Epoch 50/100][Batch 2000/5005]	 Loss 2.558, Top 1-error 35.904, Top 5-error 15.323
Train with distillation: [Epoch 50/100][Batch 2500/5005]	 Loss 2.560, Top 1-error 35.942, Top 5-error 15.354
Train with distillation: [Epoch 50/100][Batch 3000/5005]	 Loss 2.562, Top 1-error 35.974, Top 5-error 15.379
Train with distillation: [Epoch 50/100][Batch 3500/5005]	 Loss 2.566, Top 1-error 36.034, Top 5-error 15.397
Train with distillation: [Epoch 50/100][Batch 4000/5005]	 Loss 2.567, Top 1-error 36.066, Top 5-error 15.427
Train with distillation: [Epoch 50/100][Batch 4500/5005]	 Loss 2.570, Top 1-error 36.107, Top 5-error 15.449
Train with distillation: [Epoch 50/100][Batch 5000/5005]	 Loss 2.571, Top 1-error 36.111, Top 5-error 15.456
Train 	 Time Taken: 2571.78 sec
Test (on val set): [Epoch 50/100][Batch 0/196]	Time 1.856 (1.856)	Loss 0.8382 (0.8382)	Top 1-err 24.6094 (24.6094)	Top 5-err 4.2969 (4.2969)
* Epoch: [50/100]	 Top 1-err 33.024  Top 5-err 11.882	 Test Loss 1.329
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 51/100][Batch 0/5005]	 Loss 2.682, Top 1-error 38.672, Top 5-error 16.797
Train with distillation: [Epoch 51/100][Batch 500/5005]	 Loss 2.542, Top 1-error 35.683, Top 5-error 15.042
Train with distillation: [Epoch 51/100][Batch 1000/5005]	 Loss 2.554, Top 1-error 35.844, Top 5-error 15.255
Train with distillation: [Epoch 51/100][Batch 1500/5005]	 Loss 2.556, Top 1-error 35.811, Top 5-error 15.274
Train with distillation: [Epoch 51/100][Batch 2000/5005]	 Loss 2.561, Top 1-error 35.870, Top 5-error 15.323
Train with distillation: [Epoch 51/100][Batch 2500/5005]	 Loss 2.563, Top 1-error 35.932, Top 5-error 15.367
Train with distillation: [Epoch 51/100][Batch 3000/5005]	 Loss 2.566, Top 1-error 35.967, Top 5-error 15.396
Train with distillation: [Epoch 51/100][Batch 3500/5005]	 Loss 2.569, Top 1-error 36.016, Top 5-error 15.440
Train with distillation: [Epoch 51/100][Batch 4000/5005]	 Loss 2.571, Top 1-error 36.054, Top 5-error 15.450
Train with distillation: [Epoch 51/100][Batch 4500/5005]	 Loss 2.573, Top 1-error 36.075, Top 5-error 15.462
Train with distillation: [Epoch 51/100][Batch 5000/5005]	 Loss 2.574, Top 1-error 36.096, Top 5-error 15.476
Train 	 Time Taken: 2569.04 sec
Test (on val set): [Epoch 51/100][Batch 0/196]	Time 1.926 (1.926)	Loss 0.7653 (0.7653)	Top 1-err 23.8281 (23.8281)	Top 5-err 2.3438 (2.3438)
* Epoch: [51/100]	 Top 1-err 33.362  Top 5-err 12.044	 Test Loss 1.338
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 52/100][Batch 0/5005]	 Loss 2.756, Top 1-error 41.406, Top 5-error 17.578
Train with distillation: [Epoch 52/100][Batch 500/5005]	 Loss 2.545, Top 1-error 35.715, Top 5-error 15.251
Train with distillation: [Epoch 52/100][Batch 1000/5005]	 Loss 2.554, Top 1-error 35.801, Top 5-error 15.334
Train with distillation: [Epoch 52/100][Batch 1500/5005]	 Loss 2.554, Top 1-error 35.745, Top 5-error 15.296
Train with distillation: [Epoch 52/100][Batch 2000/5005]	 Loss 2.558, Top 1-error 35.784, Top 5-error 15.331
Train with distillation: [Epoch 52/100][Batch 2500/5005]	 Loss 2.557, Top 1-error 35.777, Top 5-error 15.304
Train with distillation: [Epoch 52/100][Batch 3000/5005]	 Loss 2.560, Top 1-error 35.836, Top 5-error 15.350
Train with distillation: [Epoch 52/100][Batch 3500/5005]	 Loss 2.563, Top 1-error 35.908, Top 5-error 15.387
Train with distillation: [Epoch 52/100][Batch 4000/5005]	 Loss 2.566, Top 1-error 35.952, Top 5-error 15.409
Train with distillation: [Epoch 52/100][Batch 4500/5005]	 Loss 2.568, Top 1-error 35.972, Top 5-error 15.433
Train with distillation: [Epoch 52/100][Batch 5000/5005]	 Loss 2.570, Top 1-error 36.006, Top 5-error 15.452
Train 	 Time Taken: 2562.92 sec
Test (on val set): [Epoch 52/100][Batch 0/196]	Time 1.698 (1.698)	Loss 0.7529 (0.7529)	Top 1-err 21.4844 (21.4844)	Top 5-err 3.1250 (3.1250)
* Epoch: [52/100]	 Top 1-err 32.516  Top 5-err 11.692	 Test Loss 1.313
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 53/100][Batch 0/5005]	 Loss 2.856, Top 1-error 42.578, Top 5-error 19.141
Train with distillation: [Epoch 53/100][Batch 500/5005]	 Loss 2.549, Top 1-error 35.651, Top 5-error 15.243
Train with distillation: [Epoch 53/100][Batch 1000/5005]	 Loss 2.558, Top 1-error 35.738, Top 5-error 15.360
Train with distillation: [Epoch 53/100][Batch 1500/5005]	 Loss 2.562, Top 1-error 35.833, Top 5-error 15.350
Train with distillation: [Epoch 53/100][Batch 2000/5005]	 Loss 2.561, Top 1-error 35.839, Top 5-error 15.318
Train with distillation: [Epoch 53/100][Batch 2500/5005]	 Loss 2.562, Top 1-error 35.855, Top 5-error 15.336
Train with distillation: [Epoch 53/100][Batch 3000/5005]	 Loss 2.564, Top 1-error 35.899, Top 5-error 15.359
Train with distillation: [Epoch 53/100][Batch 3500/5005]	 Loss 2.565, Top 1-error 35.920, Top 5-error 15.374
Train with distillation: [Epoch 53/100][Batch 4000/5005]	 Loss 2.568, Top 1-error 35.971, Top 5-error 15.425
Train with distillation: [Epoch 53/100][Batch 4500/5005]	 Loss 2.571, Top 1-error 36.020, Top 5-error 15.461
Train with distillation: [Epoch 53/100][Batch 5000/5005]	 Loss 2.573, Top 1-error 36.060, Top 5-error 15.479
Train 	 Time Taken: 2557.10 sec
Test (on val set): [Epoch 53/100][Batch 0/196]	Time 1.689 (1.689)	Loss 0.7234 (0.7234)	Top 1-err 23.4375 (23.4375)	Top 5-err 2.7344 (2.7344)
* Epoch: [53/100]	 Top 1-err 33.000  Top 5-err 12.000	 Test Loss 1.328
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 54/100][Batch 0/5005]	 Loss 2.605, Top 1-error 38.281, Top 5-error 15.625
Train with distillation: [Epoch 54/100][Batch 500/5005]	 Loss 2.552, Top 1-error 35.895, Top 5-error 15.215
Train with distillation: [Epoch 54/100][Batch 1000/5005]	 Loss 2.555, Top 1-error 35.780, Top 5-error 15.264
Train with distillation: [Epoch 54/100][Batch 1500/5005]	 Loss 2.560, Top 1-error 35.853, Top 5-error 15.317
Train with distillation: [Epoch 54/100][Batch 2000/5005]	 Loss 2.563, Top 1-error 35.890, Top 5-error 15.380
Train with distillation: [Epoch 54/100][Batch 2500/5005]	 Loss 2.567, Top 1-error 36.006, Top 5-error 15.440
Train with distillation: [Epoch 54/100][Batch 3000/5005]	 Loss 2.568, Top 1-error 36.019, Top 5-error 15.432
Train with distillation: [Epoch 54/100][Batch 3500/5005]	 Loss 2.568, Top 1-error 36.019, Top 5-error 15.437
Train with distillation: [Epoch 54/100][Batch 4000/5005]	 Loss 2.571, Top 1-error 36.072, Top 5-error 15.477
Train with distillation: [Epoch 54/100][Batch 4500/5005]	 Loss 2.572, Top 1-error 36.082, Top 5-error 15.483
Train with distillation: [Epoch 54/100][Batch 5000/5005]	 Loss 2.575, Top 1-error 36.117, Top 5-error 15.508
Train 	 Time Taken: 2547.60 sec
Test (on val set): [Epoch 54/100][Batch 0/196]	Time 1.720 (1.720)	Loss 0.8137 (0.8137)	Top 1-err 23.4375 (23.4375)	Top 5-err 5.8594 (5.8594)
* Epoch: [54/100]	 Top 1-err 32.908  Top 5-err 11.922	 Test Loss 1.325
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 55/100][Batch 0/5005]	 Loss 2.378, Top 1-error 32.422, Top 5-error 14.453
Train with distillation: [Epoch 55/100][Batch 500/5005]	 Loss 2.549, Top 1-error 35.709, Top 5-error 15.228
Train with distillation: [Epoch 55/100][Batch 1000/5005]	 Loss 2.553, Top 1-error 35.764, Top 5-error 15.291
Train with distillation: [Epoch 55/100][Batch 1500/5005]	 Loss 2.561, Top 1-error 35.892, Top 5-error 15.356
Train with distillation: [Epoch 55/100][Batch 2000/5005]	 Loss 2.560, Top 1-error 35.845, Top 5-error 15.353
Train with distillation: [Epoch 55/100][Batch 2500/5005]	 Loss 2.562, Top 1-error 35.848, Top 5-error 15.393
Train with distillation: [Epoch 55/100][Batch 3000/5005]	 Loss 2.563, Top 1-error 35.843, Top 5-error 15.392
Train with distillation: [Epoch 55/100][Batch 3500/5005]	 Loss 2.565, Top 1-error 35.896, Top 5-error 15.416
Train with distillation: [Epoch 55/100][Batch 4000/5005]	 Loss 2.568, Top 1-error 35.945, Top 5-error 15.442
Train with distillation: [Epoch 55/100][Batch 4500/5005]	 Loss 2.570, Top 1-error 35.980, Top 5-error 15.463
Train with distillation: [Epoch 55/100][Batch 5000/5005]	 Loss 2.572, Top 1-error 36.018, Top 5-error 15.475
Train 	 Time Taken: 2545.45 sec
Test (on val set): [Epoch 55/100][Batch 0/196]	Time 2.005 (2.005)	Loss 0.7561 (0.7561)	Top 1-err 24.2188 (24.2188)	Top 5-err 3.9062 (3.9062)
* Epoch: [55/100]	 Top 1-err 33.514  Top 5-err 12.214	 Test Loss 1.350
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 56/100][Batch 0/5005]	 Loss 2.553, Top 1-error 32.031, Top 5-error 16.406
Train with distillation: [Epoch 56/100][Batch 500/5005]	 Loss 2.554, Top 1-error 35.773, Top 5-error 15.393
Train with distillation: [Epoch 56/100][Batch 1000/5005]	 Loss 2.548, Top 1-error 35.639, Top 5-error 15.218
Train with distillation: [Epoch 56/100][Batch 1500/5005]	 Loss 2.551, Top 1-error 35.631, Top 5-error 15.215
Train with distillation: [Epoch 56/100][Batch 2000/5005]	 Loss 2.552, Top 1-error 35.692, Top 5-error 15.236
Train with distillation: [Epoch 56/100][Batch 2500/5005]	 Loss 2.556, Top 1-error 35.786, Top 5-error 15.289
Train with distillation: [Epoch 56/100][Batch 3000/5005]	 Loss 2.558, Top 1-error 35.829, Top 5-error 15.293
Train with distillation: [Epoch 56/100][Batch 3500/5005]	 Loss 2.561, Top 1-error 35.910, Top 5-error 15.345
Train with distillation: [Epoch 56/100][Batch 4000/5005]	 Loss 2.565, Top 1-error 35.965, Top 5-error 15.368
Train with distillation: [Epoch 56/100][Batch 4500/5005]	 Loss 2.568, Top 1-error 36.011, Top 5-error 15.410
Train with distillation: [Epoch 56/100][Batch 5000/5005]	 Loss 2.569, Top 1-error 36.025, Top 5-error 15.416
Train 	 Time Taken: 2540.07 sec
Test (on val set): [Epoch 56/100][Batch 0/196]	Time 1.759 (1.759)	Loss 0.7474 (0.7474)	Top 1-err 23.4375 (23.4375)	Top 5-err 2.7344 (2.7344)
* Epoch: [56/100]	 Top 1-err 33.168  Top 5-err 12.094	 Test Loss 1.333
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 57/100][Batch 0/5005]	 Loss 2.560, Top 1-error 34.766, Top 5-error 16.797
Train with distillation: [Epoch 57/100][Batch 500/5005]	 Loss 2.546, Top 1-error 35.628, Top 5-error 15.188
Train with distillation: [Epoch 57/100][Batch 1000/5005]	 Loss 2.553, Top 1-error 35.816, Top 5-error 15.227
Train with distillation: [Epoch 57/100][Batch 1500/5005]	 Loss 2.554, Top 1-error 35.794, Top 5-error 15.223
Train with distillation: [Epoch 57/100][Batch 2000/5005]	 Loss 2.555, Top 1-error 35.800, Top 5-error 15.217
Train with distillation: [Epoch 57/100][Batch 2500/5005]	 Loss 2.557, Top 1-error 35.830, Top 5-error 15.278
Train with distillation: [Epoch 57/100][Batch 3000/5005]	 Loss 2.558, Top 1-error 35.857, Top 5-error 15.299
Train with distillation: [Epoch 57/100][Batch 3500/5005]	 Loss 2.561, Top 1-error 35.899, Top 5-error 15.335
Train with distillation: [Epoch 57/100][Batch 4000/5005]	 Loss 2.563, Top 1-error 35.927, Top 5-error 15.353
Train with distillation: [Epoch 57/100][Batch 4500/5005]	 Loss 2.565, Top 1-error 35.950, Top 5-error 15.378
Train with distillation: [Epoch 57/100][Batch 5000/5005]	 Loss 2.568, Top 1-error 35.991, Top 5-error 15.408
Train 	 Time Taken: 2548.48 sec
Test (on val set): [Epoch 57/100][Batch 0/196]	Time 1.790 (1.790)	Loss 0.7058 (0.7058)	Top 1-err 21.4844 (21.4844)	Top 5-err 3.5156 (3.5156)
* Epoch: [57/100]	 Top 1-err 32.728  Top 5-err 11.718	 Test Loss 1.313
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 58/100][Batch 0/5005]	 Loss 2.376, Top 1-error 32.031, Top 5-error 12.500
Train with distillation: [Epoch 58/100][Batch 500/5005]	 Loss 2.547, Top 1-error 35.553, Top 5-error 15.031
Train with distillation: [Epoch 58/100][Batch 1000/5005]	 Loss 2.550, Top 1-error 35.567, Top 5-error 15.096
Train with distillation: [Epoch 58/100][Batch 1500/5005]	 Loss 2.553, Top 1-error 35.654, Top 5-error 15.210
Train with distillation: [Epoch 58/100][Batch 2000/5005]	 Loss 2.556, Top 1-error 35.740, Top 5-error 15.257
Train with distillation: [Epoch 58/100][Batch 2500/5005]	 Loss 2.558, Top 1-error 35.798, Top 5-error 15.283
Train with distillation: [Epoch 58/100][Batch 3000/5005]	 Loss 2.561, Top 1-error 35.857, Top 5-error 15.312
Train with distillation: [Epoch 58/100][Batch 3500/5005]	 Loss 2.562, Top 1-error 35.875, Top 5-error 15.317
Train with distillation: [Epoch 58/100][Batch 4000/5005]	 Loss 2.564, Top 1-error 35.887, Top 5-error 15.339
Train with distillation: [Epoch 58/100][Batch 4500/5005]	 Loss 2.567, Top 1-error 35.953, Top 5-error 15.379
Train with distillation: [Epoch 58/100][Batch 5000/5005]	 Loss 2.568, Top 1-error 35.983, Top 5-error 15.399
Train 	 Time Taken: 2548.51 sec
Test (on val set): [Epoch 58/100][Batch 0/196]	Time 1.764 (1.764)	Loss 0.7723 (0.7723)	Top 1-err 20.7031 (20.7031)	Top 5-err 4.6875 (4.6875)
* Epoch: [58/100]	 Top 1-err 32.542  Top 5-err 11.700	 Test Loss 1.314
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 59/100][Batch 0/5005]	 Loss 2.473, Top 1-error 36.719, Top 5-error 13.672
Train with distillation: [Epoch 59/100][Batch 500/5005]	 Loss 2.549, Top 1-error 35.792, Top 5-error 15.400
Train with distillation: [Epoch 59/100][Batch 1000/5005]	 Loss 2.557, Top 1-error 35.873, Top 5-error 15.373
Train with distillation: [Epoch 59/100][Batch 1500/5005]	 Loss 2.552, Top 1-error 35.775, Top 5-error 15.278
Train with distillation: [Epoch 59/100][Batch 2000/5005]	 Loss 2.558, Top 1-error 35.888, Top 5-error 15.316
Train with distillation: [Epoch 59/100][Batch 2500/5005]	 Loss 2.559, Top 1-error 35.923, Top 5-error 15.326
Train with distillation: [Epoch 59/100][Batch 3000/5005]	 Loss 2.561, Top 1-error 35.930, Top 5-error 15.338
Train with distillation: [Epoch 59/100][Batch 3500/5005]	 Loss 2.563, Top 1-error 35.968, Top 5-error 15.348
Train with distillation: [Epoch 59/100][Batch 4000/5005]	 Loss 2.564, Top 1-error 35.978, Top 5-error 15.353
Train with distillation: [Epoch 59/100][Batch 4500/5005]	 Loss 2.563, Top 1-error 35.960, Top 5-error 15.352
Train with distillation: [Epoch 59/100][Batch 5000/5005]	 Loss 2.566, Top 1-error 35.998, Top 5-error 15.382
Train 	 Time Taken: 2543.15 sec
Test (on val set): [Epoch 59/100][Batch 0/196]	Time 1.841 (1.841)	Loss 0.7301 (0.7301)	Top 1-err 21.4844 (21.4844)	Top 5-err 5.0781 (5.0781)
* Epoch: [59/100]	 Top 1-err 33.222  Top 5-err 12.054	 Test Loss 1.334
Current best accuracy (top-1 and 5 error): 32.306 11.46
Train with distillation: [Epoch 60/100][Batch 0/5005]	 Loss 2.698, Top 1-error 37.500, Top 5-error 16.797
Train with distillation: [Epoch 60/100][Batch 500/5005]	 Loss 2.400, Top 1-error 34.191, Top 5-error 14.360
Train with distillation: [Epoch 60/100][Batch 1000/5005]	 Loss 2.367, Top 1-error 33.839, Top 5-error 14.179
Train with distillation: [Epoch 60/100][Batch 1500/5005]	 Loss 2.348, Top 1-error 33.598, Top 5-error 14.037
Train with distillation: [Epoch 60/100][Batch 2000/5005]	 Loss 2.337, Top 1-error 33.470, Top 5-error 14.002
Train with distillation: [Epoch 60/100][Batch 2500/5005]	 Loss 2.328, Top 1-error 33.381, Top 5-error 13.921
Train with distillation: [Epoch 60/100][Batch 3000/5005]	 Loss 2.321, Top 1-error 33.295, Top 5-error 13.894
Train with distillation: [Epoch 60/100][Batch 3500/5005]	 Loss 2.316, Top 1-error 33.248, Top 5-error 13.871
Train with distillation: [Epoch 60/100][Batch 4000/5005]	 Loss 2.311, Top 1-error 33.200, Top 5-error 13.836
Train with distillation: [Epoch 60/100][Batch 4500/5005]	 Loss 2.307, Top 1-error 33.176, Top 5-error 13.813
Train with distillation: [Epoch 60/100][Batch 5000/5005]	 Loss 2.303, Top 1-error 33.139, Top 5-error 13.786
Train 	 Time Taken: 2539.78 sec
Test (on val set): [Epoch 60/100][Batch 0/196]	Time 1.779 (1.779)	Loss 0.6656 (0.6656)	Top 1-err 21.0938 (21.0938)	Top 5-err 2.7344 (2.7344)
* Epoch: [60/100]	 Top 1-err 30.018  Top 5-err 10.138	 Test Loss 1.197
Current best accuracy (top-1 and 5 error): 30.018 10.138
Train with distillation: [Epoch 61/100][Batch 0/5005]	 Loss 1.921, Top 1-error 30.469, Top 5-error 7.031
Train with distillation: [Epoch 61/100][Batch 500/5005]	 Loss 2.252, Top 1-error 32.574, Top 5-error 13.346
Train with distillation: [Epoch 61/100][Batch 1000/5005]	 Loss 2.253, Top 1-error 32.586, Top 5-error 13.405
Train with distillation: [Epoch 61/100][Batch 1500/5005]	 Loss 2.255, Top 1-error 32.658, Top 5-error 13.458
Train with distillation: [Epoch 61/100][Batch 2000/5005]	 Loss 2.255, Top 1-error 32.656, Top 5-error 13.471
Train with distillation: [Epoch 61/100][Batch 2500/5005]	 Loss 2.253, Top 1-error 32.637, Top 5-error 13.456
Train with distillation: [Epoch 61/100][Batch 3000/5005]	 Loss 2.254, Top 1-error 32.650, Top 5-error 13.471
Train with distillation: [Epoch 61/100][Batch 3500/5005]	 Loss 2.252, Top 1-error 32.616, Top 5-error 13.460
Train with distillation: [Epoch 61/100][Batch 4000/5005]	 Loss 2.251, Top 1-error 32.594, Top 5-error 13.450
Train with distillation: [Epoch 61/100][Batch 4500/5005]	 Loss 2.250, Top 1-error 32.579, Top 5-error 13.439
Train with distillation: [Epoch 61/100][Batch 5000/5005]	 Loss 2.249, Top 1-error 32.573, Top 5-error 13.433
Train 	 Time Taken: 2535.23 sec
Test (on val set): [Epoch 61/100][Batch 0/196]	Time 1.854 (1.854)	Loss 0.6539 (0.6539)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.1250 (3.1250)
* Epoch: [61/100]	 Top 1-err 29.846  Top 5-err 10.048	 Test Loss 1.188
Current best accuracy (top-1 and 5 error): 29.846 10.048
Train with distillation: [Epoch 62/100][Batch 0/5005]	 Loss 2.131, Top 1-error 27.734, Top 5-error 14.062
Train with distillation: [Epoch 62/100][Batch 500/5005]	 Loss 2.231, Top 1-error 32.320, Top 5-error 13.252
Train with distillation: [Epoch 62/100][Batch 1000/5005]	 Loss 2.231, Top 1-error 32.281, Top 5-error 13.208
Train with distillation: [Epoch 62/100][Batch 1500/5005]	 Loss 2.230, Top 1-error 32.261, Top 5-error 13.220
Train with distillation: [Epoch 62/100][Batch 2000/5005]	 Loss 2.228, Top 1-error 32.216, Top 5-error 13.232
Train with distillation: [Epoch 62/100][Batch 2500/5005]	 Loss 2.229, Top 1-error 32.289, Top 5-error 13.229
Train with distillation: [Epoch 62/100][Batch 3000/5005]	 Loss 2.229, Top 1-error 32.277, Top 5-error 13.234
Train with distillation: [Epoch 62/100][Batch 3500/5005]	 Loss 2.227, Top 1-error 32.252, Top 5-error 13.223
Train with distillation: [Epoch 62/100][Batch 4000/5005]	 Loss 2.227, Top 1-error 32.241, Top 5-error 13.230
Train with distillation: [Epoch 62/100][Batch 4500/5005]	 Loss 2.228, Top 1-error 32.250, Top 5-error 13.244
Train with distillation: [Epoch 62/100][Batch 5000/5005]	 Loss 2.226, Top 1-error 32.247, Top 5-error 13.243
Train 	 Time Taken: 2537.69 sec
Test (on val set): [Epoch 62/100][Batch 0/196]	Time 1.811 (1.811)	Loss 0.6594 (0.6594)	Top 1-err 18.3594 (18.3594)	Top 5-err 3.9062 (3.9062)
* Epoch: [62/100]	 Top 1-err 29.640  Top 5-err 10.012	 Test Loss 1.183
Current best accuracy (top-1 and 5 error): 29.64 10.012
Train with distillation: [Epoch 63/100][Batch 0/5005]	 Loss 2.233, Top 1-error 31.641, Top 5-error 14.062
Train with distillation: [Epoch 63/100][Batch 500/5005]	 Loss 2.214, Top 1-error 31.992, Top 5-error 13.166
Train with distillation: [Epoch 63/100][Batch 1000/5005]	 Loss 2.213, Top 1-error 32.085, Top 5-error 13.147
Train with distillation: [Epoch 63/100][Batch 1500/5005]	 Loss 2.208, Top 1-error 31.993, Top 5-error 13.070
Train with distillation: [Epoch 63/100][Batch 2000/5005]	 Loss 2.208, Top 1-error 31.990, Top 5-error 13.064
Train with distillation: [Epoch 63/100][Batch 2500/5005]	 Loss 2.209, Top 1-error 31.982, Top 5-error 13.086
Train with distillation: [Epoch 63/100][Batch 3000/5005]	 Loss 2.208, Top 1-error 31.989, Top 5-error 13.085
Train with distillation: [Epoch 63/100][Batch 3500/5005]	 Loss 2.209, Top 1-error 32.005, Top 5-error 13.111
Train with distillation: [Epoch 63/100][Batch 4000/5005]	 Loss 2.210, Top 1-error 32.038, Top 5-error 13.107
Train with distillation: [Epoch 63/100][Batch 4500/5005]	 Loss 2.210, Top 1-error 32.045, Top 5-error 13.103
Train with distillation: [Epoch 63/100][Batch 5000/5005]	 Loss 2.210, Top 1-error 32.052, Top 5-error 13.107
Train 	 Time Taken: 2537.77 sec
Test (on val set): [Epoch 63/100][Batch 0/196]	Time 1.963 (1.963)	Loss 0.6490 (0.6490)	Top 1-err 18.7500 (18.7500)	Top 5-err 2.7344 (2.7344)
* Epoch: [63/100]	 Top 1-err 29.526  Top 5-err 9.976	 Test Loss 1.176
Current best accuracy (top-1 and 5 error): 29.526 9.976
Train with distillation: [Epoch 64/100][Batch 0/5005]	 Loss 2.051, Top 1-error 28.516, Top 5-error 13.672
Train with distillation: [Epoch 64/100][Batch 500/5005]	 Loss 2.200, Top 1-error 31.868, Top 5-error 13.042
Train with distillation: [Epoch 64/100][Batch 1000/5005]	 Loss 2.199, Top 1-error 31.913, Top 5-error 13.008
Train with distillation: [Epoch 64/100][Batch 1500/5005]	 Loss 2.201, Top 1-error 31.905, Top 5-error 13.054
Train with distillation: [Epoch 64/100][Batch 2000/5005]	 Loss 2.201, Top 1-error 31.938, Top 5-error 13.050
Train with distillation: [Epoch 64/100][Batch 2500/5005]	 Loss 2.199, Top 1-error 31.890, Top 5-error 13.038
Train with distillation: [Epoch 64/100][Batch 3000/5005]	 Loss 2.199, Top 1-error 31.880, Top 5-error 13.046
Train with distillation: [Epoch 64/100][Batch 3500/5005]	 Loss 2.199, Top 1-error 31.896, Top 5-error 13.023
Train with distillation: [Epoch 64/100][Batch 4000/5005]	 Loss 2.199, Top 1-error 31.898, Top 5-error 13.019
Train with distillation: [Epoch 64/100][Batch 4500/5005]	 Loss 2.199, Top 1-error 31.905, Top 5-error 13.017
Train with distillation: [Epoch 64/100][Batch 5000/5005]	 Loss 2.200, Top 1-error 31.928, Top 5-error 13.037
Train 	 Time Taken: 2535.46 sec
Test (on val set): [Epoch 64/100][Batch 0/196]	Time 1.824 (1.824)	Loss 0.6362 (0.6362)	Top 1-err 18.3594 (18.3594)	Top 5-err 2.7344 (2.7344)
* Epoch: [64/100]	 Top 1-err 29.482  Top 5-err 9.884	 Test Loss 1.172
Current best accuracy (top-1 and 5 error): 29.482 9.884
Train with distillation: [Epoch 65/100][Batch 0/5005]	 Loss 2.216, Top 1-error 34.375, Top 5-error 12.109
Train with distillation: [Epoch 65/100][Batch 500/5005]	 Loss 2.187, Top 1-error 31.814, Top 5-error 12.884
Train with distillation: [Epoch 65/100][Batch 1000/5005]	 Loss 2.192, Top 1-error 31.811, Top 5-error 12.968
Train with distillation: [Epoch 65/100][Batch 1500/5005]	 Loss 2.194, Top 1-error 31.851, Top 5-error 12.981
Train with distillation: [Epoch 65/100][Batch 2000/5005]	 Loss 2.195, Top 1-error 31.810, Top 5-error 13.009
Train with distillation: [Epoch 65/100][Batch 2500/5005]	 Loss 2.194, Top 1-error 31.793, Top 5-error 13.008
Train with distillation: [Epoch 65/100][Batch 3000/5005]	 Loss 2.194, Top 1-error 31.807, Top 5-error 12.986
Train with distillation: [Epoch 65/100][Batch 3500/5005]	 Loss 2.193, Top 1-error 31.792, Top 5-error 12.966
Train with distillation: [Epoch 65/100][Batch 4000/5005]	 Loss 2.193, Top 1-error 31.795, Top 5-error 12.971
Train with distillation: [Epoch 65/100][Batch 4500/5005]	 Loss 2.194, Top 1-error 31.813, Top 5-error 12.987
Train with distillation: [Epoch 65/100][Batch 5000/5005]	 Loss 2.195, Top 1-error 31.825, Top 5-error 13.002
Train 	 Time Taken: 2527.95 sec
Test (on val set): [Epoch 65/100][Batch 0/196]	Time 1.676 (1.676)	Loss 0.6431 (0.6431)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.1250 (3.1250)
* Epoch: [65/100]	 Top 1-err 29.522  Top 5-err 9.844	 Test Loss 1.172
Current best accuracy (top-1 and 5 error): 29.482 9.884
Train with distillation: [Epoch 66/100][Batch 0/5005]	 Loss 2.049, Top 1-error 31.641, Top 5-error 10.938
Train with distillation: [Epoch 66/100][Batch 500/5005]	 Loss 2.186, Top 1-error 31.821, Top 5-error 12.926
Train with distillation: [Epoch 66/100][Batch 1000/5005]	 Loss 2.183, Top 1-error 31.723, Top 5-error 12.864
Train with distillation: [Epoch 66/100][Batch 1500/5005]	 Loss 2.184, Top 1-error 31.692, Top 5-error 12.871
Train with distillation: [Epoch 66/100][Batch 2000/5005]	 Loss 2.183, Top 1-error 31.702, Top 5-error 12.896
Train with distillation: [Epoch 66/100][Batch 2500/5005]	 Loss 2.182, Top 1-error 31.685, Top 5-error 12.882
Train with distillation: [Epoch 66/100][Batch 3000/5005]	 Loss 2.184, Top 1-error 31.722, Top 5-error 12.915
Train with distillation: [Epoch 66/100][Batch 3500/5005]	 Loss 2.187, Top 1-error 31.782, Top 5-error 12.952
Train with distillation: [Epoch 66/100][Batch 4000/5005]	 Loss 2.187, Top 1-error 31.763, Top 5-error 12.952
Train with distillation: [Epoch 66/100][Batch 4500/5005]	 Loss 2.187, Top 1-error 31.762, Top 5-error 12.948
Train with distillation: [Epoch 66/100][Batch 5000/5005]	 Loss 2.188, Top 1-error 31.793, Top 5-error 12.967
Train 	 Time Taken: 2526.78 sec
Test (on val set): [Epoch 66/100][Batch 0/196]	Time 1.866 (1.866)	Loss 0.6542 (0.6542)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.9062 (3.9062)
* Epoch: [66/100]	 Top 1-err 29.354  Top 5-err 9.836	 Test Loss 1.169
Current best accuracy (top-1 and 5 error): 29.354 9.836
Train with distillation: [Epoch 67/100][Batch 0/5005]	 Loss 2.313, Top 1-error 30.859, Top 5-error 11.719
Train with distillation: [Epoch 67/100][Batch 500/5005]	 Loss 2.178, Top 1-error 31.632, Top 5-error 12.735
Train with distillation: [Epoch 67/100][Batch 1000/5005]	 Loss 2.178, Top 1-error 31.643, Top 5-error 12.761
Train with distillation: [Epoch 67/100][Batch 1500/5005]	 Loss 2.179, Top 1-error 31.698, Top 5-error 12.818
Train with distillation: [Epoch 67/100][Batch 2000/5005]	 Loss 2.180, Top 1-error 31.689, Top 5-error 12.828
Train with distillation: [Epoch 67/100][Batch 2500/5005]	 Loss 2.180, Top 1-error 31.675, Top 5-error 12.831
Train with distillation: [Epoch 67/100][Batch 3000/5005]	 Loss 2.180, Top 1-error 31.689, Top 5-error 12.839
Train with distillation: [Epoch 67/100][Batch 3500/5005]	 Loss 2.179, Top 1-error 31.679, Top 5-error 12.833
Train with distillation: [Epoch 67/100][Batch 4000/5005]	 Loss 2.180, Top 1-error 31.694, Top 5-error 12.845
Train with distillation: [Epoch 67/100][Batch 4500/5005]	 Loss 2.180, Top 1-error 31.688, Top 5-error 12.850
Train with distillation: [Epoch 67/100][Batch 5000/5005]	 Loss 2.180, Top 1-error 31.690, Top 5-error 12.850
Train 	 Time Taken: 2520.35 sec
Test (on val set): [Epoch 67/100][Batch 0/196]	Time 1.831 (1.831)	Loss 0.6523 (0.6523)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.1250 (3.1250)
* Epoch: [67/100]	 Top 1-err 29.448  Top 5-err 9.794	 Test Loss 1.169
Current best accuracy (top-1 and 5 error): 29.354 9.836
Train with distillation: [Epoch 68/100][Batch 0/5005]	 Loss 1.936, Top 1-error 27.734, Top 5-error 10.938
Train with distillation: [Epoch 68/100][Batch 500/5005]	 Loss 2.162, Top 1-error 31.383, Top 5-error 12.620
Train with distillation: [Epoch 68/100][Batch 1000/5005]	 Loss 2.164, Top 1-error 31.360, Top 5-error 12.612
Train with distillation: [Epoch 68/100][Batch 1500/5005]	 Loss 2.169, Top 1-error 31.437, Top 5-error 12.715
Train with distillation: [Epoch 68/100][Batch 2000/5005]	 Loss 2.171, Top 1-error 31.478, Top 5-error 12.754
Train with distillation: [Epoch 68/100][Batch 2500/5005]	 Loss 2.172, Top 1-error 31.551, Top 5-error 12.794
Train with distillation: [Epoch 68/100][Batch 3000/5005]	 Loss 2.172, Top 1-error 31.546, Top 5-error 12.798
Train with distillation: [Epoch 68/100][Batch 3500/5005]	 Loss 2.173, Top 1-error 31.570, Top 5-error 12.820
Train with distillation: [Epoch 68/100][Batch 4000/5005]	 Loss 2.175, Top 1-error 31.601, Top 5-error 12.838
Train with distillation: [Epoch 68/100][Batch 4500/5005]	 Loss 2.175, Top 1-error 31.606, Top 5-error 12.840
Train with distillation: [Epoch 68/100][Batch 5000/5005]	 Loss 2.175, Top 1-error 31.613, Top 5-error 12.841
Train 	 Time Taken: 2517.90 sec
Test (on val set): [Epoch 68/100][Batch 0/196]	Time 1.798 (1.798)	Loss 0.6294 (0.6294)	Top 1-err 18.7500 (18.7500)	Top 5-err 2.7344 (2.7344)
* Epoch: [68/100]	 Top 1-err 29.236  Top 5-err 9.810	 Test Loss 1.165
Current best accuracy (top-1 and 5 error): 29.236 9.81
Train with distillation: [Epoch 69/100][Batch 0/5005]	 Loss 2.351, Top 1-error 33.984, Top 5-error 13.672
Train with distillation: [Epoch 69/100][Batch 500/5005]	 Loss 2.165, Top 1-error 31.433, Top 5-error 12.840
Train with distillation: [Epoch 69/100][Batch 1000/5005]	 Loss 2.168, Top 1-error 31.413, Top 5-error 12.814
Train with distillation: [Epoch 69/100][Batch 1500/5005]	 Loss 2.171, Top 1-error 31.573, Top 5-error 12.832
Train with distillation: [Epoch 69/100][Batch 2000/5005]	 Loss 2.170, Top 1-error 31.569, Top 5-error 12.824
Train with distillation: [Epoch 69/100][Batch 2500/5005]	 Loss 2.172, Top 1-error 31.564, Top 5-error 12.822
Train with distillation: [Epoch 69/100][Batch 3000/5005]	 Loss 2.172, Top 1-error 31.569, Top 5-error 12.831
Train with distillation: [Epoch 69/100][Batch 3500/5005]	 Loss 2.172, Top 1-error 31.569, Top 5-error 12.810
Train with distillation: [Epoch 69/100][Batch 4000/5005]	 Loss 2.172, Top 1-error 31.572, Top 5-error 12.799
Train with distillation: [Epoch 69/100][Batch 4500/5005]	 Loss 2.173, Top 1-error 31.583, Top 5-error 12.807
Train with distillation: [Epoch 69/100][Batch 5000/5005]	 Loss 2.172, Top 1-error 31.588, Top 5-error 12.810
Train 	 Time Taken: 2508.22 sec
Test (on val set): [Epoch 69/100][Batch 0/196]	Time 1.805 (1.805)	Loss 0.6316 (0.6316)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.5156 (3.5156)
* Epoch: [69/100]	 Top 1-err 29.236  Top 5-err 9.770	 Test Loss 1.160
Current best accuracy (top-1 and 5 error): 29.236 9.77
Train with distillation: [Epoch 70/100][Batch 0/5005]	 Loss 2.264, Top 1-error 36.719, Top 5-error 10.156
Train with distillation: [Epoch 70/100][Batch 500/5005]	 Loss 2.159, Top 1-error 31.404, Top 5-error 12.630
Train with distillation: [Epoch 70/100][Batch 1000/5005]	 Loss 2.159, Top 1-error 31.363, Top 5-error 12.612
Train with distillation: [Epoch 70/100][Batch 1500/5005]	 Loss 2.160, Top 1-error 31.405, Top 5-error 12.634
Train with distillation: [Epoch 70/100][Batch 2000/5005]	 Loss 2.164, Top 1-error 31.488, Top 5-error 12.690
Train with distillation: [Epoch 70/100][Batch 2500/5005]	 Loss 2.166, Top 1-error 31.531, Top 5-error 12.730
Train with distillation: [Epoch 70/100][Batch 3000/5005]	 Loss 2.166, Top 1-error 31.542, Top 5-error 12.745
Train with distillation: [Epoch 70/100][Batch 3500/5005]	 Loss 2.167, Top 1-error 31.538, Top 5-error 12.746
Train with distillation: [Epoch 70/100][Batch 4000/5005]	 Loss 2.167, Top 1-error 31.547, Top 5-error 12.752
Train with distillation: [Epoch 70/100][Batch 4500/5005]	 Loss 2.168, Top 1-error 31.572, Top 5-error 12.768
Train with distillation: [Epoch 70/100][Batch 5000/5005]	 Loss 2.168, Top 1-error 31.579, Top 5-error 12.781
Train 	 Time Taken: 2511.35 sec
Test (on val set): [Epoch 70/100][Batch 0/196]	Time 1.812 (1.812)	Loss 0.6147 (0.6147)	Top 1-err 18.3594 (18.3594)	Top 5-err 2.3438 (2.3438)
* Epoch: [70/100]	 Top 1-err 29.338  Top 5-err 9.808	 Test Loss 1.164
Current best accuracy (top-1 and 5 error): 29.236 9.77
Train with distillation: [Epoch 71/100][Batch 0/5005]	 Loss 2.070, Top 1-error 27.734, Top 5-error 13.672
Train with distillation: [Epoch 71/100][Batch 500/5005]	 Loss 2.152, Top 1-error 31.229, Top 5-error 12.619
Train with distillation: [Epoch 71/100][Batch 1000/5005]	 Loss 2.160, Top 1-error 31.376, Top 5-error 12.728
Train with distillation: [Epoch 71/100][Batch 1500/5005]	 Loss 2.161, Top 1-error 31.373, Top 5-error 12.737
Train with distillation: [Epoch 71/100][Batch 2000/5005]	 Loss 2.161, Top 1-error 31.371, Top 5-error 12.725
Train with distillation: [Epoch 71/100][Batch 2500/5005]	 Loss 2.162, Top 1-error 31.421, Top 5-error 12.714
Train with distillation: [Epoch 71/100][Batch 3000/5005]	 Loss 2.164, Top 1-error 31.453, Top 5-error 12.736
Train with distillation: [Epoch 71/100][Batch 3500/5005]	 Loss 2.164, Top 1-error 31.465, Top 5-error 12.741
Train with distillation: [Epoch 71/100][Batch 4000/5005]	 Loss 2.164, Top 1-error 31.470, Top 5-error 12.746
Train with distillation: [Epoch 71/100][Batch 4500/5005]	 Loss 2.164, Top 1-error 31.452, Top 5-error 12.742
Train with distillation: [Epoch 71/100][Batch 5000/5005]	 Loss 2.164, Top 1-error 31.442, Top 5-error 12.734
Train 	 Time Taken: 2512.81 sec
Test (on val set): [Epoch 71/100][Batch 0/196]	Time 1.730 (1.730)	Loss 0.6415 (0.6415)	Top 1-err 21.0938 (21.0938)	Top 5-err 3.1250 (3.1250)
* Epoch: [71/100]	 Top 1-err 29.230  Top 5-err 9.704	 Test Loss 1.161
Current best accuracy (top-1 and 5 error): 29.23 9.704
Train with distillation: [Epoch 72/100][Batch 0/5005]	 Loss 2.104, Top 1-error 30.859, Top 5-error 13.281
Train with distillation: [Epoch 72/100][Batch 500/5005]	 Loss 2.158, Top 1-error 31.437, Top 5-error 12.600
Train with distillation: [Epoch 72/100][Batch 1000/5005]	 Loss 2.157, Top 1-error 31.319, Top 5-error 12.627
Train with distillation: [Epoch 72/100][Batch 1500/5005]	 Loss 2.157, Top 1-error 31.335, Top 5-error 12.645
Train with distillation: [Epoch 72/100][Batch 2000/5005]	 Loss 2.156, Top 1-error 31.311, Top 5-error 12.640
Train with distillation: [Epoch 72/100][Batch 2500/5005]	 Loss 2.158, Top 1-error 31.361, Top 5-error 12.671
Train with distillation: [Epoch 72/100][Batch 3000/5005]	 Loss 2.158, Top 1-error 31.361, Top 5-error 12.675
Train with distillation: [Epoch 72/100][Batch 3500/5005]	 Loss 2.160, Top 1-error 31.395, Top 5-error 12.718
Train with distillation: [Epoch 72/100][Batch 4000/5005]	 Loss 2.161, Top 1-error 31.411, Top 5-error 12.736
Train with distillation: [Epoch 72/100][Batch 4500/5005]	 Loss 2.162, Top 1-error 31.425, Top 5-error 12.738
Train with distillation: [Epoch 72/100][Batch 5000/5005]	 Loss 2.162, Top 1-error 31.430, Top 5-error 12.734
Train 	 Time Taken: 2509.85 sec
Test (on val set): [Epoch 72/100][Batch 0/196]	Time 2.048 (2.048)	Loss 0.6272 (0.6272)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.1250 (3.1250)
* Epoch: [72/100]	 Top 1-err 29.214  Top 5-err 9.720	 Test Loss 1.162
Current best accuracy (top-1 and 5 error): 29.214 9.72
Train with distillation: [Epoch 73/100][Batch 0/5005]	 Loss 2.083, Top 1-error 29.297, Top 5-error 12.109
Train with distillation: [Epoch 73/100][Batch 500/5005]	 Loss 2.146, Top 1-error 31.128, Top 5-error 12.470
Train with distillation: [Epoch 73/100][Batch 1000/5005]	 Loss 2.153, Top 1-error 31.248, Top 5-error 12.559
Train with distillation: [Epoch 73/100][Batch 1500/5005]	 Loss 2.155, Top 1-error 31.298, Top 5-error 12.604
Train with distillation: [Epoch 73/100][Batch 2000/5005]	 Loss 2.153, Top 1-error 31.274, Top 5-error 12.600
Train with distillation: [Epoch 73/100][Batch 2500/5005]	 Loss 2.153, Top 1-error 31.303, Top 5-error 12.597
Train with distillation: [Epoch 73/100][Batch 3000/5005]	 Loss 2.154, Top 1-error 31.321, Top 5-error 12.618
Train with distillation: [Epoch 73/100][Batch 3500/5005]	 Loss 2.154, Top 1-error 31.298, Top 5-error 12.632
Train with distillation: [Epoch 73/100][Batch 4000/5005]	 Loss 2.155, Top 1-error 31.322, Top 5-error 12.648
Train with distillation: [Epoch 73/100][Batch 4500/5005]	 Loss 2.156, Top 1-error 31.363, Top 5-error 12.672
Train with distillation: [Epoch 73/100][Batch 5000/5005]	 Loss 2.157, Top 1-error 31.372, Top 5-error 12.683
Train 	 Time Taken: 2506.69 sec
Test (on val set): [Epoch 73/100][Batch 0/196]	Time 1.611 (1.611)	Loss 0.6038 (0.6038)	Top 1-err 17.9688 (17.9688)	Top 5-err 3.5156 (3.5156)
* Epoch: [73/100]	 Top 1-err 29.212  Top 5-err 9.694	 Test Loss 1.159
Current best accuracy (top-1 and 5 error): 29.212 9.694
Train with distillation: [Epoch 74/100][Batch 0/5005]	 Loss 2.131, Top 1-error 28.125, Top 5-error 14.062
Train with distillation: [Epoch 74/100][Batch 500/5005]	 Loss 2.153, Top 1-error 31.317, Top 5-error 12.728
Train with distillation: [Epoch 74/100][Batch 1000/5005]	 Loss 2.152, Top 1-error 31.341, Top 5-error 12.679
Train with distillation: [Epoch 74/100][Batch 1500/5005]	 Loss 2.155, Top 1-error 31.391, Top 5-error 12.675
Train with distillation: [Epoch 74/100][Batch 2000/5005]	 Loss 2.158, Top 1-error 31.397, Top 5-error 12.725
Train with distillation: [Epoch 74/100][Batch 2500/5005]	 Loss 2.158, Top 1-error 31.376, Top 5-error 12.706
Train with distillation: [Epoch 74/100][Batch 3000/5005]	 Loss 2.157, Top 1-error 31.362, Top 5-error 12.694
Train with distillation: [Epoch 74/100][Batch 3500/5005]	 Loss 2.156, Top 1-error 31.352, Top 5-error 12.679
Train with distillation: [Epoch 74/100][Batch 4000/5005]	 Loss 2.155, Top 1-error 31.340, Top 5-error 12.659
Train with distillation: [Epoch 74/100][Batch 4500/5005]	 Loss 2.155, Top 1-error 31.326, Top 5-error 12.653
Train with distillation: [Epoch 74/100][Batch 5000/5005]	 Loss 2.155, Top 1-error 31.317, Top 5-error 12.655
Train 	 Time Taken: 2505.01 sec
Test (on val set): [Epoch 74/100][Batch 0/196]	Time 1.641 (1.641)	Loss 0.6390 (0.6390)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.1250 (3.1250)
* Epoch: [74/100]	 Top 1-err 29.164  Top 5-err 9.710	 Test Loss 1.160
Current best accuracy (top-1 and 5 error): 29.164 9.71
Train with distillation: [Epoch 75/100][Batch 0/5005]	 Loss 2.056, Top 1-error 33.203, Top 5-error 12.891
Train with distillation: [Epoch 75/100][Batch 500/5005]	 Loss 2.146, Top 1-error 31.221, Top 5-error 12.599
Train with distillation: [Epoch 75/100][Batch 1000/5005]	 Loss 2.148, Top 1-error 31.214, Top 5-error 12.639
Train with distillation: [Epoch 75/100][Batch 1500/5005]	 Loss 2.151, Top 1-error 31.271, Top 5-error 12.650
Train with distillation: [Epoch 75/100][Batch 2000/5005]	 Loss 2.150, Top 1-error 31.287, Top 5-error 12.629
Train with distillation: [Epoch 75/100][Batch 2500/5005]	 Loss 2.152, Top 1-error 31.324, Top 5-error 12.647
Train with distillation: [Epoch 75/100][Batch 3000/5005]	 Loss 2.153, Top 1-error 31.331, Top 5-error 12.672
Train with distillation: [Epoch 75/100][Batch 3500/5005]	 Loss 2.154, Top 1-error 31.338, Top 5-error 12.677
Train with distillation: [Epoch 75/100][Batch 4000/5005]	 Loss 2.153, Top 1-error 31.309, Top 5-error 12.671
Train with distillation: [Epoch 75/100][Batch 4500/5005]	 Loss 2.153, Top 1-error 31.304, Top 5-error 12.654
Train with distillation: [Epoch 75/100][Batch 5000/5005]	 Loss 2.153, Top 1-error 31.311, Top 5-error 12.652
Train 	 Time Taken: 2506.23 sec
Test (on val set): [Epoch 75/100][Batch 0/196]	Time 1.729 (1.729)	Loss 0.6083 (0.6083)	Top 1-err 17.5781 (17.5781)	Top 5-err 2.3438 (2.3438)
* Epoch: [75/100]	 Top 1-err 29.010  Top 5-err 9.642	 Test Loss 1.156
Current best accuracy (top-1 and 5 error): 29.01 9.642
Train with distillation: [Epoch 76/100][Batch 0/5005]	 Loss 2.331, Top 1-error 36.328, Top 5-error 16.797
Train with distillation: [Epoch 76/100][Batch 500/5005]	 Loss 2.147, Top 1-error 31.170, Top 5-error 12.593
Train with distillation: [Epoch 76/100][Batch 1000/5005]	 Loss 2.148, Top 1-error 31.228, Top 5-error 12.614
Train with distillation: [Epoch 76/100][Batch 1500/5005]	 Loss 2.146, Top 1-error 31.159, Top 5-error 12.550
Train with distillation: [Epoch 76/100][Batch 2000/5005]	 Loss 2.148, Top 1-error 31.176, Top 5-error 12.589
Train with distillation: [Epoch 76/100][Batch 2500/5005]	 Loss 2.146, Top 1-error 31.152, Top 5-error 12.564
Train with distillation: [Epoch 76/100][Batch 3000/5005]	 Loss 2.147, Top 1-error 31.179, Top 5-error 12.597
Train with distillation: [Epoch 76/100][Batch 3500/5005]	 Loss 2.147, Top 1-error 31.193, Top 5-error 12.601
Train with distillation: [Epoch 76/100][Batch 4000/5005]	 Loss 2.149, Top 1-error 31.231, Top 5-error 12.615
Train with distillation: [Epoch 76/100][Batch 4500/5005]	 Loss 2.149, Top 1-error 31.227, Top 5-error 12.622
Train with distillation: [Epoch 76/100][Batch 5000/5005]	 Loss 2.150, Top 1-error 31.248, Top 5-error 12.632
Train 	 Time Taken: 2503.56 sec
Test (on val set): [Epoch 76/100][Batch 0/196]	Time 1.610 (1.610)	Loss 0.6576 (0.6576)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.5156 (3.5156)
* Epoch: [76/100]	 Top 1-err 29.176  Top 5-err 9.650	 Test Loss 1.157
Current best accuracy (top-1 and 5 error): 29.01 9.642
Train with distillation: [Epoch 77/100][Batch 0/5005]	 Loss 1.968, Top 1-error 28.516, Top 5-error 10.938
Train with distillation: [Epoch 77/100][Batch 500/5005]	 Loss 2.139, Top 1-error 31.042, Top 5-error 12.478
Train with distillation: [Epoch 77/100][Batch 1000/5005]	 Loss 2.139, Top 1-error 31.054, Top 5-error 12.491
Train with distillation: [Epoch 77/100][Batch 1500/5005]	 Loss 2.138, Top 1-error 31.005, Top 5-error 12.494
Train with distillation: [Epoch 77/100][Batch 2000/5005]	 Loss 2.141, Top 1-error 31.076, Top 5-error 12.506
Train with distillation: [Epoch 77/100][Batch 2500/5005]	 Loss 2.143, Top 1-error 31.117, Top 5-error 12.530
Train with distillation: [Epoch 77/100][Batch 3000/5005]	 Loss 2.143, Top 1-error 31.123, Top 5-error 12.540
Train with distillation: [Epoch 77/100][Batch 3500/5005]	 Loss 2.144, Top 1-error 31.131, Top 5-error 12.549
Train with distillation: [Epoch 77/100][Batch 4000/5005]	 Loss 2.144, Top 1-error 31.124, Top 5-error 12.543
Train with distillation: [Epoch 77/100][Batch 4500/5005]	 Loss 2.145, Top 1-error 31.145, Top 5-error 12.558
Train with distillation: [Epoch 77/100][Batch 5000/5005]	 Loss 2.146, Top 1-error 31.161, Top 5-error 12.570
Train 	 Time Taken: 2500.72 sec
Test (on val set): [Epoch 77/100][Batch 0/196]	Time 1.860 (1.860)	Loss 0.6453 (0.6453)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.9062 (3.9062)
* Epoch: [77/100]	 Top 1-err 28.996  Top 5-err 9.644	 Test Loss 1.154
Current best accuracy (top-1 and 5 error): 28.996 9.644
Train with distillation: [Epoch 78/100][Batch 0/5005]	 Loss 2.405, Top 1-error 34.375, Top 5-error 17.969
Train with distillation: [Epoch 78/100][Batch 500/5005]	 Loss 2.137, Top 1-error 31.026, Top 5-error 12.482
Train with distillation: [Epoch 78/100][Batch 1000/5005]	 Loss 2.141, Top 1-error 31.065, Top 5-error 12.535
Train with distillation: [Epoch 78/100][Batch 1500/5005]	 Loss 2.143, Top 1-error 31.090, Top 5-error 12.576
Train with distillation: [Epoch 78/100][Batch 2000/5005]	 Loss 2.144, Top 1-error 31.106, Top 5-error 12.573
Train with distillation: [Epoch 78/100][Batch 2500/5005]	 Loss 2.143, Top 1-error 31.114, Top 5-error 12.559
Train with distillation: [Epoch 78/100][Batch 3000/5005]	 Loss 2.143, Top 1-error 31.092, Top 5-error 12.551
Train with distillation: [Epoch 78/100][Batch 3500/5005]	 Loss 2.144, Top 1-error 31.090, Top 5-error 12.570
Train with distillation: [Epoch 78/100][Batch 4000/5005]	 Loss 2.145, Top 1-error 31.133, Top 5-error 12.589
Train with distillation: [Epoch 78/100][Batch 4500/5005]	 Loss 2.145, Top 1-error 31.144, Top 5-error 12.580
Train with distillation: [Epoch 78/100][Batch 5000/5005]	 Loss 2.146, Top 1-error 31.144, Top 5-error 12.585
Train 	 Time Taken: 2503.05 sec
Test (on val set): [Epoch 78/100][Batch 0/196]	Time 1.698 (1.698)	Loss 0.6558 (0.6558)	Top 1-err 21.0938 (21.0938)	Top 5-err 3.1250 (3.1250)
* Epoch: [78/100]	 Top 1-err 29.078  Top 5-err 9.740	 Test Loss 1.156
Current best accuracy (top-1 and 5 error): 28.996 9.644
Train with distillation: [Epoch 79/100][Batch 0/5005]	 Loss 2.197, Top 1-error 32.031, Top 5-error 12.500
Train with distillation: [Epoch 79/100][Batch 500/5005]	 Loss 2.133, Top 1-error 31.024, Top 5-error 12.445
Train with distillation: [Epoch 79/100][Batch 1000/5005]	 Loss 2.133, Top 1-error 31.024, Top 5-error 12.492
Train with distillation: [Epoch 79/100][Batch 1500/5005]	 Loss 2.134, Top 1-error 31.018, Top 5-error 12.482
Train with distillation: [Epoch 79/100][Batch 2000/5005]	 Loss 2.136, Top 1-error 31.047, Top 5-error 12.494
Train with distillation: [Epoch 79/100][Batch 2500/5005]	 Loss 2.138, Top 1-error 31.067, Top 5-error 12.502
Train with distillation: [Epoch 79/100][Batch 3000/5005]	 Loss 2.138, Top 1-error 31.072, Top 5-error 12.488
Train with distillation: [Epoch 79/100][Batch 3500/5005]	 Loss 2.140, Top 1-error 31.111, Top 5-error 12.512
Train with distillation: [Epoch 79/100][Batch 4000/5005]	 Loss 2.141, Top 1-error 31.123, Top 5-error 12.514
Train with distillation: [Epoch 79/100][Batch 4500/5005]	 Loss 2.141, Top 1-error 31.110, Top 5-error 12.520
Train with distillation: [Epoch 79/100][Batch 5000/5005]	 Loss 2.142, Top 1-error 31.120, Top 5-error 12.543
Train 	 Time Taken: 2498.14 sec
Test (on val set): [Epoch 79/100][Batch 0/196]	Time 1.835 (1.835)	Loss 0.6127 (0.6127)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.1250 (3.1250)
* Epoch: [79/100]	 Top 1-err 28.978  Top 5-err 9.630	 Test Loss 1.152
Current best accuracy (top-1 and 5 error): 28.978 9.63
Train with distillation: [Epoch 80/100][Batch 0/5005]	 Loss 1.910, Top 1-error 24.609, Top 5-error 9.766
Train with distillation: [Epoch 80/100][Batch 500/5005]	 Loss 2.131, Top 1-error 30.969, Top 5-error 12.449
Train with distillation: [Epoch 80/100][Batch 1000/5005]	 Loss 2.134, Top 1-error 30.949, Top 5-error 12.490
Train with distillation: [Epoch 80/100][Batch 1500/5005]	 Loss 2.136, Top 1-error 30.991, Top 5-error 12.504
Train with distillation: [Epoch 80/100][Batch 2000/5005]	 Loss 2.135, Top 1-error 30.973, Top 5-error 12.493
Train with distillation: [Epoch 80/100][Batch 2500/5005]	 Loss 2.135, Top 1-error 30.966, Top 5-error 12.470
Train with distillation: [Epoch 80/100][Batch 3000/5005]	 Loss 2.135, Top 1-error 30.958, Top 5-error 12.464
Train with distillation: [Epoch 80/100][Batch 3500/5005]	 Loss 2.136, Top 1-error 30.968, Top 5-error 12.463
Train with distillation: [Epoch 80/100][Batch 4000/5005]	 Loss 2.137, Top 1-error 31.009, Top 5-error 12.459
Train with distillation: [Epoch 80/100][Batch 4500/5005]	 Loss 2.137, Top 1-error 31.033, Top 5-error 12.471
Train with distillation: [Epoch 80/100][Batch 5000/5005]	 Loss 2.137, Top 1-error 31.045, Top 5-error 12.483
Train 	 Time Taken: 2493.63 sec
Test (on val set): [Epoch 80/100][Batch 0/196]	Time 1.637 (1.637)	Loss 0.6596 (0.6596)	Top 1-err 21.0938 (21.0938)	Top 5-err 3.5156 (3.5156)
* Epoch: [80/100]	 Top 1-err 29.106  Top 5-err 9.620	 Test Loss 1.152
Current best accuracy (top-1 and 5 error): 28.978 9.63
Train with distillation: [Epoch 81/100][Batch 0/5005]	 Loss 2.287, Top 1-error 33.984, Top 5-error 14.844
Train with distillation: [Epoch 81/100][Batch 500/5005]	 Loss 2.127, Top 1-error 30.909, Top 5-error 12.448
Train with distillation: [Epoch 81/100][Batch 1000/5005]	 Loss 2.131, Top 1-error 30.882, Top 5-error 12.454
Train with distillation: [Epoch 81/100][Batch 1500/5005]	 Loss 2.133, Top 1-error 30.946, Top 5-error 12.459
Train with distillation: [Epoch 81/100][Batch 2000/5005]	 Loss 2.133, Top 1-error 30.979, Top 5-error 12.448
Train with distillation: [Epoch 81/100][Batch 2500/5005]	 Loss 2.134, Top 1-error 30.973, Top 5-error 12.470
Train with distillation: [Epoch 81/100][Batch 3000/5005]	 Loss 2.136, Top 1-error 30.989, Top 5-error 12.494
Train with distillation: [Epoch 81/100][Batch 3500/5005]	 Loss 2.137, Top 1-error 31.011, Top 5-error 12.510
Train with distillation: [Epoch 81/100][Batch 4000/5005]	 Loss 2.137, Top 1-error 31.005, Top 5-error 12.517
Train with distillation: [Epoch 81/100][Batch 4500/5005]	 Loss 2.137, Top 1-error 31.028, Top 5-error 12.510
Train with distillation: [Epoch 81/100][Batch 5000/5005]	 Loss 2.138, Top 1-error 31.035, Top 5-error 12.514
Train 	 Time Taken: 2478.91 sec
Test (on val set): [Epoch 81/100][Batch 0/196]	Time 1.704 (1.704)	Loss 0.6574 (0.6574)	Top 1-err 19.5312 (19.5312)	Top 5-err 3.1250 (3.1250)
* Epoch: [81/100]	 Top 1-err 29.044  Top 5-err 9.646	 Test Loss 1.153
Current best accuracy (top-1 and 5 error): 28.978 9.63
Train with distillation: [Epoch 82/100][Batch 0/5005]	 Loss 2.309, Top 1-error 39.062, Top 5-error 14.062
Train with distillation: [Epoch 82/100][Batch 500/5005]	 Loss 2.131, Top 1-error 30.841, Top 5-error 12.469
Train with distillation: [Epoch 82/100][Batch 1000/5005]	 Loss 2.133, Top 1-error 30.910, Top 5-error 12.470
Train with distillation: [Epoch 82/100][Batch 1500/5005]	 Loss 2.133, Top 1-error 30.949, Top 5-error 12.480
Train with distillation: [Epoch 82/100][Batch 2000/5005]	 Loss 2.133, Top 1-error 30.925, Top 5-error 12.467
Train with distillation: [Epoch 82/100][Batch 2500/5005]	 Loss 2.132, Top 1-error 30.901, Top 5-error 12.438
Train with distillation: [Epoch 82/100][Batch 3000/5005]	 Loss 2.133, Top 1-error 30.934, Top 5-error 12.454
Train with distillation: [Epoch 82/100][Batch 3500/5005]	 Loss 2.135, Top 1-error 30.961, Top 5-error 12.471
Train with distillation: [Epoch 82/100][Batch 4000/5005]	 Loss 2.136, Top 1-error 30.982, Top 5-error 12.482
Train with distillation: [Epoch 82/100][Batch 4500/5005]	 Loss 2.136, Top 1-error 30.994, Top 5-error 12.493
Train with distillation: [Epoch 82/100][Batch 5000/5005]	 Loss 2.137, Top 1-error 31.018, Top 5-error 12.505
Train 	 Time Taken: 2479.23 sec
Test (on val set): [Epoch 82/100][Batch 0/196]	Time 1.608 (1.608)	Loss 0.6337 (0.6337)	Top 1-err 19.5312 (19.5312)	Top 5-err 3.5156 (3.5156)
* Epoch: [82/100]	 Top 1-err 29.096  Top 5-err 9.530	 Test Loss 1.151
Current best accuracy (top-1 and 5 error): 28.978 9.63
Train with distillation: [Epoch 83/100][Batch 0/5005]	 Loss 1.984, Top 1-error 27.344, Top 5-error 11.719
Train with distillation: [Epoch 83/100][Batch 500/5005]	 Loss 2.125, Top 1-error 30.741, Top 5-error 12.344
Train with distillation: [Epoch 83/100][Batch 1000/5005]	 Loss 2.125, Top 1-error 30.769, Top 5-error 12.363
Train with distillation: [Epoch 83/100][Batch 1500/5005]	 Loss 2.128, Top 1-error 30.842, Top 5-error 12.380
Train with distillation: [Epoch 83/100][Batch 2000/5005]	 Loss 2.130, Top 1-error 30.903, Top 5-error 12.407
Train with distillation: [Epoch 83/100][Batch 2500/5005]	 Loss 2.130, Top 1-error 30.893, Top 5-error 12.412
Train with distillation: [Epoch 83/100][Batch 3000/5005]	 Loss 2.131, Top 1-error 30.911, Top 5-error 12.427
Train with distillation: [Epoch 83/100][Batch 3500/5005]	 Loss 2.132, Top 1-error 30.931, Top 5-error 12.443
Train with distillation: [Epoch 83/100][Batch 4000/5005]	 Loss 2.133, Top 1-error 30.956, Top 5-error 12.444
Train with distillation: [Epoch 83/100][Batch 4500/5005]	 Loss 2.134, Top 1-error 30.971, Top 5-error 12.455
Train with distillation: [Epoch 83/100][Batch 5000/5005]	 Loss 2.134, Top 1-error 30.970, Top 5-error 12.453
Train 	 Time Taken: 2477.22 sec
Test (on val set): [Epoch 83/100][Batch 0/196]	Time 1.863 (1.863)	Loss 0.6254 (0.6254)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.5156 (3.5156)
* Epoch: [83/100]	 Top 1-err 28.972  Top 5-err 9.618	 Test Loss 1.152
Current best accuracy (top-1 and 5 error): 28.972 9.618
Train with distillation: [Epoch 84/100][Batch 0/5005]	 Loss 2.049, Top 1-error 32.031, Top 5-error 10.156
Train with distillation: [Epoch 84/100][Batch 500/5005]	 Loss 2.120, Top 1-error 30.768, Top 5-error 12.215
Train with distillation: [Epoch 84/100][Batch 1000/5005]	 Loss 2.124, Top 1-error 30.792, Top 5-error 12.304
Train with distillation: [Epoch 84/100][Batch 1500/5005]	 Loss 2.125, Top 1-error 30.874, Top 5-error 12.348
Train with distillation: [Epoch 84/100][Batch 2000/5005]	 Loss 2.127, Top 1-error 30.928, Top 5-error 12.375
Train with distillation: [Epoch 84/100][Batch 2500/5005]	 Loss 2.128, Top 1-error 30.900, Top 5-error 12.411
Train with distillation: [Epoch 84/100][Batch 3000/5005]	 Loss 2.129, Top 1-error 30.914, Top 5-error 12.423
Train with distillation: [Epoch 84/100][Batch 3500/5005]	 Loss 2.130, Top 1-error 30.922, Top 5-error 12.448
Train with distillation: [Epoch 84/100][Batch 4000/5005]	 Loss 2.132, Top 1-error 30.944, Top 5-error 12.462
Train with distillation: [Epoch 84/100][Batch 4500/5005]	 Loss 2.133, Top 1-error 30.980, Top 5-error 12.482
Train with distillation: [Epoch 84/100][Batch 5000/5005]	 Loss 2.134, Top 1-error 30.993, Top 5-error 12.482
Train 	 Time Taken: 2487.83 sec
Test (on val set): [Epoch 84/100][Batch 0/196]	Time 1.695 (1.695)	Loss 0.6288 (0.6288)	Top 1-err 18.3594 (18.3594)	Top 5-err 3.1250 (3.1250)
* Epoch: [84/100]	 Top 1-err 28.986  Top 5-err 9.680	 Test Loss 1.153
Current best accuracy (top-1 and 5 error): 28.972 9.618
Train with distillation: [Epoch 85/100][Batch 0/5005]	 Loss 2.002, Top 1-error 28.906, Top 5-error 12.500
Train with distillation: [Epoch 85/100][Batch 500/5005]	 Loss 2.125, Top 1-error 30.769, Top 5-error 12.448
Train with distillation: [Epoch 85/100][Batch 1000/5005]	 Loss 2.123, Top 1-error 30.775, Top 5-error 12.349
Train with distillation: [Epoch 85/100][Batch 1500/5005]	 Loss 2.123, Top 1-error 30.754, Top 5-error 12.332
Train with distillation: [Epoch 85/100][Batch 2000/5005]	 Loss 2.126, Top 1-error 30.830, Top 5-error 12.385
Train with distillation: [Epoch 85/100][Batch 2500/5005]	 Loss 2.125, Top 1-error 30.768, Top 5-error 12.376
Train with distillation: [Epoch 85/100][Batch 3000/5005]	 Loss 2.126, Top 1-error 30.780, Top 5-error 12.404
Train with distillation: [Epoch 85/100][Batch 3500/5005]	 Loss 2.129, Top 1-error 30.832, Top 5-error 12.422
Train with distillation: [Epoch 85/100][Batch 4000/5005]	 Loss 2.129, Top 1-error 30.856, Top 5-error 12.422
Train with distillation: [Epoch 85/100][Batch 4500/5005]	 Loss 2.130, Top 1-error 30.871, Top 5-error 12.429
Train with distillation: [Epoch 85/100][Batch 5000/5005]	 Loss 2.131, Top 1-error 30.889, Top 5-error 12.441
Train 	 Time Taken: 2485.49 sec
Test (on val set): [Epoch 85/100][Batch 0/196]	Time 1.812 (1.812)	Loss 0.6248 (0.6248)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.5156 (3.5156)
* Epoch: [85/100]	 Top 1-err 28.938  Top 5-err 9.640	 Test Loss 1.149
Current best accuracy (top-1 and 5 error): 28.938 9.64
Train with distillation: [Epoch 86/100][Batch 0/5005]	 Loss 2.149, Top 1-error 30.078, Top 5-error 11.719
Train with distillation: [Epoch 86/100][Batch 500/5005]	 Loss 2.126, Top 1-error 30.755, Top 5-error 12.349
Train with distillation: [Epoch 86/100][Batch 1000/5005]	 Loss 2.123, Top 1-error 30.735, Top 5-error 12.273
Train with distillation: [Epoch 86/100][Batch 1500/5005]	 Loss 2.126, Top 1-error 30.814, Top 5-error 12.346
Train with distillation: [Epoch 86/100][Batch 2000/5005]	 Loss 2.126, Top 1-error 30.846, Top 5-error 12.349
Train with distillation: [Epoch 86/100][Batch 2500/5005]	 Loss 2.126, Top 1-error 30.851, Top 5-error 12.365
Train with distillation: [Epoch 86/100][Batch 3000/5005]	 Loss 2.128, Top 1-error 30.870, Top 5-error 12.376
Train with distillation: [Epoch 86/100][Batch 3500/5005]	 Loss 2.128, Top 1-error 30.862, Top 5-error 12.379
Train with distillation: [Epoch 86/100][Batch 4000/5005]	 Loss 2.129, Top 1-error 30.896, Top 5-error 12.393
Train with distillation: [Epoch 86/100][Batch 4500/5005]	 Loss 2.129, Top 1-error 30.912, Top 5-error 12.398
Train with distillation: [Epoch 86/100][Batch 5000/5005]	 Loss 2.129, Top 1-error 30.914, Top 5-error 12.390
Train 	 Time Taken: 2489.56 sec
Test (on val set): [Epoch 86/100][Batch 0/196]	Time 1.662 (1.662)	Loss 0.6682 (0.6682)	Top 1-err 19.5312 (19.5312)	Top 5-err 3.9062 (3.9062)
* Epoch: [86/100]	 Top 1-err 28.920  Top 5-err 9.688	 Test Loss 1.151
Current best accuracy (top-1 and 5 error): 28.92 9.688
Train with distillation: [Epoch 87/100][Batch 0/5005]	 Loss 2.201, Top 1-error 30.859, Top 5-error 13.672
Train with distillation: [Epoch 87/100][Batch 500/5005]	 Loss 2.115, Top 1-error 30.575, Top 5-error 12.282
Train with distillation: [Epoch 87/100][Batch 1000/5005]	 Loss 2.121, Top 1-error 30.720, Top 5-error 12.345
Train with distillation: [Epoch 87/100][Batch 1500/5005]	 Loss 2.124, Top 1-error 30.820, Top 5-error 12.409
Train with distillation: [Epoch 87/100][Batch 2000/5005]	 Loss 2.126, Top 1-error 30.842, Top 5-error 12.411
Train with distillation: [Epoch 87/100][Batch 2500/5005]	 Loss 2.126, Top 1-error 30.847, Top 5-error 12.402
Train with distillation: [Epoch 87/100][Batch 3000/5005]	 Loss 2.127, Top 1-error 30.839, Top 5-error 12.400
Train with distillation: [Epoch 87/100][Batch 3500/5005]	 Loss 2.127, Top 1-error 30.850, Top 5-error 12.403
Train with distillation: [Epoch 87/100][Batch 4000/5005]	 Loss 2.128, Top 1-error 30.871, Top 5-error 12.405
Train with distillation: [Epoch 87/100][Batch 4500/5005]	 Loss 2.128, Top 1-error 30.850, Top 5-error 12.401
Train with distillation: [Epoch 87/100][Batch 5000/5005]	 Loss 2.129, Top 1-error 30.861, Top 5-error 12.417
Train 	 Time Taken: 2482.88 sec
Test (on val set): [Epoch 87/100][Batch 0/196]	Time 1.926 (1.926)	Loss 0.6122 (0.6122)	Top 1-err 18.7500 (18.7500)	Top 5-err 2.7344 (2.7344)
* Epoch: [87/100]	 Top 1-err 28.834  Top 5-err 9.576	 Test Loss 1.148
Current best accuracy (top-1 and 5 error): 28.834 9.576
Train with distillation: [Epoch 88/100][Batch 0/5005]	 Loss 2.076, Top 1-error 29.297, Top 5-error 14.453
Train with distillation: [Epoch 88/100][Batch 500/5005]	 Loss 2.123, Top 1-error 30.724, Top 5-error 12.485
Train with distillation: [Epoch 88/100][Batch 1000/5005]	 Loss 2.125, Top 1-error 30.793, Top 5-error 12.449
Train with distillation: [Epoch 88/100][Batch 1500/5005]	 Loss 2.125, Top 1-error 30.796, Top 5-error 12.403
Train with distillation: [Epoch 88/100][Batch 2000/5005]	 Loss 2.128, Top 1-error 30.861, Top 5-error 12.419
Train with distillation: [Epoch 88/100][Batch 2500/5005]	 Loss 2.127, Top 1-error 30.874, Top 5-error 12.406
Train with distillation: [Epoch 88/100][Batch 3000/5005]	 Loss 2.129, Top 1-error 30.924, Top 5-error 12.435
Train with distillation: [Epoch 88/100][Batch 3500/5005]	 Loss 2.129, Top 1-error 30.911, Top 5-error 12.434
Train with distillation: [Epoch 88/100][Batch 4000/5005]	 Loss 2.129, Top 1-error 30.903, Top 5-error 12.443
Train with distillation: [Epoch 88/100][Batch 4500/5005]	 Loss 2.128, Top 1-error 30.882, Top 5-error 12.415
Train with distillation: [Epoch 88/100][Batch 5000/5005]	 Loss 2.127, Top 1-error 30.871, Top 5-error 12.406
Train 	 Time Taken: 2488.69 sec
Test (on val set): [Epoch 88/100][Batch 0/196]	Time 1.675 (1.675)	Loss 0.6538 (0.6538)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.5156 (3.5156)
* Epoch: [88/100]	 Top 1-err 28.886  Top 5-err 9.566	 Test Loss 1.149
Current best accuracy (top-1 and 5 error): 28.834 9.576
Train with distillation: [Epoch 89/100][Batch 0/5005]	 Loss 2.076, Top 1-error 28.125, Top 5-error 11.328
Train with distillation: [Epoch 89/100][Batch 500/5005]	 Loss 2.117, Top 1-error 30.628, Top 5-error 12.197
Train with distillation: [Epoch 89/100][Batch 1000/5005]	 Loss 2.118, Top 1-error 30.658, Top 5-error 12.224
Train with distillation: [Epoch 89/100][Batch 1500/5005]	 Loss 2.122, Top 1-error 30.752, Top 5-error 12.286
Train with distillation: [Epoch 89/100][Batch 2000/5005]	 Loss 2.123, Top 1-error 30.773, Top 5-error 12.323
Train with distillation: [Epoch 89/100][Batch 2500/5005]	 Loss 2.124, Top 1-error 30.756, Top 5-error 12.353
Train with distillation: [Epoch 89/100][Batch 3000/5005]	 Loss 2.124, Top 1-error 30.760, Top 5-error 12.360
Train with distillation: [Epoch 89/100][Batch 3500/5005]	 Loss 2.125, Top 1-error 30.766, Top 5-error 12.375
Train with distillation: [Epoch 89/100][Batch 4000/5005]	 Loss 2.126, Top 1-error 30.778, Top 5-error 12.392
Train with distillation: [Epoch 89/100][Batch 4500/5005]	 Loss 2.128, Top 1-error 30.833, Top 5-error 12.423
Train with distillation: [Epoch 89/100][Batch 5000/5005]	 Loss 2.127, Top 1-error 30.837, Top 5-error 12.409
Train 	 Time Taken: 2490.55 sec
Test (on val set): [Epoch 89/100][Batch 0/196]	Time 1.882 (1.882)	Loss 0.6219 (0.6219)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.5156 (3.5156)
* Epoch: [89/100]	 Top 1-err 28.800  Top 5-err 9.626	 Test Loss 1.144
Current best accuracy (top-1 and 5 error): 28.8 9.626
Train with distillation: [Epoch 90/100][Batch 0/5005]	 Loss 2.070, Top 1-error 30.859, Top 5-error 10.547
Train with distillation: [Epoch 90/100][Batch 500/5005]	 Loss 2.106, Top 1-error 30.643, Top 5-error 12.153
Train with distillation: [Epoch 90/100][Batch 1000/5005]	 Loss 2.104, Top 1-error 30.608, Top 5-error 12.204
Train with distillation: [Epoch 90/100][Batch 1500/5005]	 Loss 2.102, Top 1-error 30.564, Top 5-error 12.206
Train with distillation: [Epoch 90/100][Batch 2000/5005]	 Loss 2.099, Top 1-error 30.487, Top 5-error 12.191
Train with distillation: [Epoch 90/100][Batch 2500/5005]	 Loss 2.098, Top 1-error 30.473, Top 5-error 12.202
Train with distillation: [Epoch 90/100][Batch 3000/5005]	 Loss 2.097, Top 1-error 30.471, Top 5-error 12.203
Train with distillation: [Epoch 90/100][Batch 3500/5005]	 Loss 2.095, Top 1-error 30.448, Top 5-error 12.171
Train with distillation: [Epoch 90/100][Batch 4000/5005]	 Loss 2.094, Top 1-error 30.426, Top 5-error 12.160
Train with distillation: [Epoch 90/100][Batch 4500/5005]	 Loss 2.093, Top 1-error 30.431, Top 5-error 12.159
Train with distillation: [Epoch 90/100][Batch 5000/5005]	 Loss 2.093, Top 1-error 30.431, Top 5-error 12.164
Train 	 Time Taken: 2493.61 sec
Test (on val set): [Epoch 90/100][Batch 0/196]	Time 1.715 (1.715)	Loss 0.6291 (0.6291)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.5156 (3.5156)
* Epoch: [90/100]	 Top 1-err 28.600  Top 5-err 9.364	 Test Loss 1.135
Current best accuracy (top-1 and 5 error): 28.6 9.364
Train with distillation: [Epoch 91/100][Batch 0/5005]	 Loss 2.306, Top 1-error 36.719, Top 5-error 16.797
Train with distillation: [Epoch 91/100][Batch 500/5005]	 Loss 2.088, Top 1-error 30.435, Top 5-error 12.201
Train with distillation: [Epoch 91/100][Batch 1000/5005]	 Loss 2.089, Top 1-error 30.416, Top 5-error 12.262
Train with distillation: [Epoch 91/100][Batch 1500/5005]	 Loss 2.090, Top 1-error 30.410, Top 5-error 12.216
Train with distillation: [Epoch 91/100][Batch 2000/5005]	 Loss 2.089, Top 1-error 30.364, Top 5-error 12.182
Train with distillation: [Epoch 91/100][Batch 2500/5005]	 Loss 2.089, Top 1-error 30.382, Top 5-error 12.199
Train with distillation: [Epoch 91/100][Batch 3000/5005]	 Loss 2.088, Top 1-error 30.372, Top 5-error 12.168
Train with distillation: [Epoch 91/100][Batch 3500/5005]	 Loss 2.088, Top 1-error 30.373, Top 5-error 12.173
Train with distillation: [Epoch 91/100][Batch 4000/5005]	 Loss 2.087, Top 1-error 30.365, Top 5-error 12.152
Train with distillation: [Epoch 91/100][Batch 4500/5005]	 Loss 2.087, Top 1-error 30.351, Top 5-error 12.152
Train with distillation: [Epoch 91/100][Batch 5000/5005]	 Loss 2.086, Top 1-error 30.330, Top 5-error 12.147
Train 	 Time Taken: 2501.33 sec
Test (on val set): [Epoch 91/100][Batch 0/196]	Time 1.704 (1.704)	Loss 0.6218 (0.6218)	Top 1-err 19.9219 (19.9219)	Top 5-err 3.5156 (3.5156)
* Epoch: [91/100]	 Top 1-err 28.552  Top 5-err 9.376	 Test Loss 1.134
Current best accuracy (top-1 and 5 error): 28.552 9.376
Train with distillation: [Epoch 92/100][Batch 0/5005]	 Loss 2.080, Top 1-error 28.125, Top 5-error 12.109
Train with distillation: [Epoch 92/100][Batch 500/5005]	 Loss 2.095, Top 1-error 30.530, Top 5-error 12.227
Train with distillation: [Epoch 92/100][Batch 1000/5005]	 Loss 2.093, Top 1-error 30.506, Top 5-error 12.216
Train with distillation: [Epoch 92/100][Batch 1500/5005]	 Loss 2.093, Top 1-error 30.489, Top 5-error 12.254
Train with distillation: [Epoch 92/100][Batch 2000/5005]	 Loss 2.089, Top 1-error 30.435, Top 5-error 12.212
Train with distillation: [Epoch 92/100][Batch 2500/5005]	 Loss 2.086, Top 1-error 30.393, Top 5-error 12.150
Train with distillation: [Epoch 92/100][Batch 3000/5005]	 Loss 2.085, Top 1-error 30.370, Top 5-error 12.133
Train with distillation: [Epoch 92/100][Batch 3500/5005]	 Loss 2.085, Top 1-error 30.341, Top 5-error 12.146
Train with distillation: [Epoch 92/100][Batch 4000/5005]	 Loss 2.084, Top 1-error 30.350, Top 5-error 12.145
Train with distillation: [Epoch 92/100][Batch 4500/5005]	 Loss 2.084, Top 1-error 30.341, Top 5-error 12.134
Train with distillation: [Epoch 92/100][Batch 5000/5005]	 Loss 2.084, Top 1-error 30.343, Top 5-error 12.140
Train 	 Time Taken: 2503.61 sec
Test (on val set): [Epoch 92/100][Batch 0/196]	Time 1.805 (1.805)	Loss 0.6118 (0.6118)	Top 1-err 18.3594 (18.3594)	Top 5-err 3.5156 (3.5156)
* Epoch: [92/100]	 Top 1-err 28.504  Top 5-err 9.414	 Test Loss 1.133
Current best accuracy (top-1 and 5 error): 28.504 9.414
Train with distillation: [Epoch 93/100][Batch 0/5005]	 Loss 2.219, Top 1-error 35.156, Top 5-error 12.109
Train with distillation: [Epoch 93/100][Batch 500/5005]	 Loss 2.086, Top 1-error 30.412, Top 5-error 12.222
Train with distillation: [Epoch 93/100][Batch 1000/5005]	 Loss 2.084, Top 1-error 30.350, Top 5-error 12.186
Train with distillation: [Epoch 93/100][Batch 1500/5005]	 Loss 2.083, Top 1-error 30.357, Top 5-error 12.183
Train with distillation: [Epoch 93/100][Batch 2000/5005]	 Loss 2.084, Top 1-error 30.324, Top 5-error 12.177
Train with distillation: [Epoch 93/100][Batch 2500/5005]	 Loss 2.082, Top 1-error 30.290, Top 5-error 12.162
Train with distillation: [Epoch 93/100][Batch 3000/5005]	 Loss 2.082, Top 1-error 30.321, Top 5-error 12.140
Train with distillation: [Epoch 93/100][Batch 3500/5005]	 Loss 2.081, Top 1-error 30.273, Top 5-error 12.107
Train with distillation: [Epoch 93/100][Batch 4000/5005]	 Loss 2.082, Top 1-error 30.315, Top 5-error 12.117
Train with distillation: [Epoch 93/100][Batch 4500/5005]	 Loss 2.081, Top 1-error 30.306, Top 5-error 12.105
Train with distillation: [Epoch 93/100][Batch 5000/5005]	 Loss 2.080, Top 1-error 30.297, Top 5-error 12.104
Train 	 Time Taken: 2502.83 sec
Test (on val set): [Epoch 93/100][Batch 0/196]	Time 1.877 (1.877)	Loss 0.6110 (0.6110)	Top 1-err 17.9688 (17.9688)	Top 5-err 3.1250 (3.1250)
* Epoch: [93/100]	 Top 1-err 28.548  Top 5-err 9.390	 Test Loss 1.134
Current best accuracy (top-1 and 5 error): 28.504 9.414
Train with distillation: [Epoch 94/100][Batch 0/5005]	 Loss 2.061, Top 1-error 28.516, Top 5-error 11.719
Train with distillation: [Epoch 94/100][Batch 500/5005]	 Loss 2.075, Top 1-error 30.273, Top 5-error 12.031
Train with distillation: [Epoch 94/100][Batch 1000/5005]	 Loss 2.074, Top 1-error 30.243, Top 5-error 12.047
Train with distillation: [Epoch 94/100][Batch 1500/5005]	 Loss 2.080, Top 1-error 30.304, Top 5-error 12.086
Train with distillation: [Epoch 94/100][Batch 2000/5005]	 Loss 2.080, Top 1-error 30.285, Top 5-error 12.067
Train with distillation: [Epoch 94/100][Batch 2500/5005]	 Loss 2.079, Top 1-error 30.272, Top 5-error 12.069
Train with distillation: [Epoch 94/100][Batch 3000/5005]	 Loss 2.079, Top 1-error 30.261, Top 5-error 12.071
Train with distillation: [Epoch 94/100][Batch 3500/5005]	 Loss 2.079, Top 1-error 30.267, Top 5-error 12.075
Train with distillation: [Epoch 94/100][Batch 4000/5005]	 Loss 2.079, Top 1-error 30.268, Top 5-error 12.074
Train with distillation: [Epoch 94/100][Batch 4500/5005]	 Loss 2.079, Top 1-error 30.284, Top 5-error 12.080
Train with distillation: [Epoch 94/100][Batch 5000/5005]	 Loss 2.078, Top 1-error 30.284, Top 5-error 12.061
Train 	 Time Taken: 2515.47 sec
Test (on val set): [Epoch 94/100][Batch 0/196]	Time 1.792 (1.792)	Loss 0.6238 (0.6238)	Top 1-err 19.5312 (19.5312)	Top 5-err 3.1250 (3.1250)
* Epoch: [94/100]	 Top 1-err 28.500  Top 5-err 9.430	 Test Loss 1.132
Current best accuracy (top-1 and 5 error): 28.5 9.43
Train with distillation: [Epoch 95/100][Batch 0/5005]	 Loss 1.857, Top 1-error 27.344, Top 5-error 8.984
Train with distillation: [Epoch 95/100][Batch 500/5005]	 Loss 2.086, Top 1-error 30.488, Top 5-error 12.197
Train with distillation: [Epoch 95/100][Batch 1000/5005]	 Loss 2.083, Top 1-error 30.391, Top 5-error 12.160
Train with distillation: [Epoch 95/100][Batch 1500/5005]	 Loss 2.082, Top 1-error 30.404, Top 5-error 12.134
Train with distillation: [Epoch 95/100][Batch 2000/5005]	 Loss 2.081, Top 1-error 30.385, Top 5-error 12.084
Train with distillation: [Epoch 95/100][Batch 2500/5005]	 Loss 2.080, Top 1-error 30.370, Top 5-error 12.064
Train with distillation: [Epoch 95/100][Batch 3000/5005]	 Loss 2.080, Top 1-error 30.345, Top 5-error 12.066
Train with distillation: [Epoch 95/100][Batch 3500/5005]	 Loss 2.079, Top 1-error 30.327, Top 5-error 12.058
Train with distillation: [Epoch 95/100][Batch 4000/5005]	 Loss 2.077, Top 1-error 30.290, Top 5-error 12.043
Train with distillation: [Epoch 95/100][Batch 4500/5005]	 Loss 2.077, Top 1-error 30.263, Top 5-error 12.029
Train with distillation: [Epoch 95/100][Batch 5000/5005]	 Loss 2.077, Top 1-error 30.260, Top 5-error 12.030
Train 	 Time Taken: 2512.46 sec
Test (on val set): [Epoch 95/100][Batch 0/196]	Time 1.952 (1.952)	Loss 0.6204 (0.6204)	Top 1-err 19.5312 (19.5312)	Top 5-err 3.5156 (3.5156)
* Epoch: [95/100]	 Top 1-err 28.444  Top 5-err 9.342	 Test Loss 1.132
Current best accuracy (top-1 and 5 error): 28.444 9.342
Train with distillation: [Epoch 96/100][Batch 0/5005]	 Loss 2.235, Top 1-error 32.422, Top 5-error 16.406
Train with distillation: [Epoch 96/100][Batch 500/5005]	 Loss 2.073, Top 1-error 30.139, Top 5-error 12.038
Train with distillation: [Epoch 96/100][Batch 1000/5005]	 Loss 2.076, Top 1-error 30.258, Top 5-error 12.074
Train with distillation: [Epoch 96/100][Batch 1500/5005]	 Loss 2.077, Top 1-error 30.256, Top 5-error 12.080
Train with distillation: [Epoch 96/100][Batch 2000/5005]	 Loss 2.076, Top 1-error 30.233, Top 5-error 12.066
Train with distillation: [Epoch 96/100][Batch 2500/5005]	 Loss 2.075, Top 1-error 30.227, Top 5-error 12.054
Train with distillation: [Epoch 96/100][Batch 3000/5005]	 Loss 2.076, Top 1-error 30.252, Top 5-error 12.049
Train with distillation: [Epoch 96/100][Batch 3500/5005]	 Loss 2.076, Top 1-error 30.254, Top 5-error 12.050
Train with distillation: [Epoch 96/100][Batch 4000/5005]	 Loss 2.075, Top 1-error 30.222, Top 5-error 12.042
Train with distillation: [Epoch 96/100][Batch 4500/5005]	 Loss 2.076, Top 1-error 30.250, Top 5-error 12.055
Train with distillation: [Epoch 96/100][Batch 5000/5005]	 Loss 2.077, Top 1-error 30.250, Top 5-error 12.057
Train 	 Time Taken: 2512.08 sec
Test (on val set): [Epoch 96/100][Batch 0/196]	Time 1.769 (1.769)	Loss 0.6190 (0.6190)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.5156 (3.5156)
* Epoch: [96/100]	 Top 1-err 28.484  Top 5-err 9.356	 Test Loss 1.133
Current best accuracy (top-1 and 5 error): 28.444 9.342
Train with distillation: [Epoch 97/100][Batch 0/5005]	 Loss 2.046, Top 1-error 31.250, Top 5-error 10.547
Train with distillation: [Epoch 97/100][Batch 500/5005]	 Loss 2.075, Top 1-error 30.206, Top 5-error 11.976
Train with distillation: [Epoch 97/100][Batch 1000/5005]	 Loss 2.074, Top 1-error 30.216, Top 5-error 12.002
Train with distillation: [Epoch 97/100][Batch 1500/5005]	 Loss 2.074, Top 1-error 30.218, Top 5-error 12.018
Train with distillation: [Epoch 97/100][Batch 2000/5005]	 Loss 2.076, Top 1-error 30.223, Top 5-error 12.058
Train with distillation: [Epoch 97/100][Batch 2500/5005]	 Loss 2.076, Top 1-error 30.229, Top 5-error 12.075
Train with distillation: [Epoch 97/100][Batch 3000/5005]	 Loss 2.079, Top 1-error 30.277, Top 5-error 12.119
Train with distillation: [Epoch 97/100][Batch 3500/5005]	 Loss 2.078, Top 1-error 30.277, Top 5-error 12.100
Train with distillation: [Epoch 97/100][Batch 4000/5005]	 Loss 2.078, Top 1-error 30.295, Top 5-error 12.094
Train with distillation: [Epoch 97/100][Batch 4500/5005]	 Loss 2.079, Top 1-error 30.290, Top 5-error 12.082
Train with distillation: [Epoch 97/100][Batch 5000/5005]	 Loss 2.078, Top 1-error 30.282, Top 5-error 12.087
Train 	 Time Taken: 2511.11 sec
Test (on val set): [Epoch 97/100][Batch 0/196]	Time 1.950 (1.950)	Loss 0.6187 (0.6187)	Top 1-err 18.7500 (18.7500)	Top 5-err 3.5156 (3.5156)
* Epoch: [97/100]	 Top 1-err 28.428  Top 5-err 9.406	 Test Loss 1.131
Current best accuracy (top-1 and 5 error): 28.428 9.406
Train with distillation: [Epoch 98/100][Batch 0/5005]	 Loss 2.100, Top 1-error 30.859, Top 5-error 11.328
Train with distillation: [Epoch 98/100][Batch 500/5005]	 Loss 2.076, Top 1-error 30.222, Top 5-error 12.029
Train with distillation: [Epoch 98/100][Batch 1000/5005]	 Loss 2.075, Top 1-error 30.255, Top 5-error 12.056
Train with distillation: [Epoch 98/100][Batch 1500/5005]	 Loss 2.076, Top 1-error 30.279, Top 5-error 12.044
Train with distillation: [Epoch 98/100][Batch 2000/5005]	 Loss 2.076, Top 1-error 30.275, Top 5-error 12.038
Train with distillation: [Epoch 98/100][Batch 2500/5005]	 Loss 2.077, Top 1-error 30.261, Top 5-error 12.045
Train with distillation: [Epoch 98/100][Batch 3000/5005]	 Loss 2.077, Top 1-error 30.250, Top 5-error 12.049
Train with distillation: [Epoch 98/100][Batch 3500/5005]	 Loss 2.078, Top 1-error 30.249, Top 5-error 12.062
Train with distillation: [Epoch 98/100][Batch 4000/5005]	 Loss 2.076, Top 1-error 30.254, Top 5-error 12.049
Train with distillation: [Epoch 98/100][Batch 4500/5005]	 Loss 2.076, Top 1-error 30.248, Top 5-error 12.047
Train with distillation: [Epoch 98/100][Batch 5000/5005]	 Loss 2.076, Top 1-error 30.236, Top 5-error 12.027
Train 	 Time Taken: 2516.89 sec
Test (on val set): [Epoch 98/100][Batch 0/196]	Time 1.763 (1.763)	Loss 0.6290 (0.6290)	Top 1-err 19.1406 (19.1406)	Top 5-err 3.5156 (3.5156)
* Epoch: [98/100]	 Top 1-err 28.530  Top 5-err 9.354	 Test Loss 1.131
Current best accuracy (top-1 and 5 error): 28.428 9.406
Train with distillation: [Epoch 99/100][Batch 0/5005]	 Loss 2.168, Top 1-error 29.297, Top 5-error 12.500
Train with distillation: [Epoch 99/100][Batch 500/5005]	 Loss 2.082, Top 1-error 30.300, Top 5-error 12.267
Train with distillation: [Epoch 99/100][Batch 1000/5005]	 Loss 2.080, Top 1-error 30.295, Top 5-error 12.224
Train with distillation: [Epoch 99/100][Batch 1500/5005]	 Loss 2.079, Top 1-error 30.290, Top 5-error 12.160
Train with distillation: [Epoch 99/100][Batch 2000/5005]	 Loss 2.077, Top 1-error 30.228, Top 5-error 12.108
Train with distillation: [Epoch 99/100][Batch 2500/5005]	 Loss 2.077, Top 1-error 30.241, Top 5-error 12.114
Train with distillation: [Epoch 99/100][Batch 3000/5005]	 Loss 2.076, Top 1-error 30.220, Top 5-error 12.092
Train with distillation: [Epoch 99/100][Batch 3500/5005]	 Loss 2.076, Top 1-error 30.211, Top 5-error 12.079
Train with distillation: [Epoch 99/100][Batch 4000/5005]	 Loss 2.077, Top 1-error 30.235, Top 5-error 12.078
Train with distillation: [Epoch 99/100][Batch 4500/5005]	 Loss 2.076, Top 1-error 30.228, Top 5-error 12.059
Train with distillation: [Epoch 99/100][Batch 5000/5005]	 Loss 2.076, Top 1-error 30.228, Top 5-error 12.059
Train 	 Time Taken: 2513.69 sec
Test (on val set): [Epoch 99/100][Batch 0/196]	Time 1.873 (1.873)	Loss 0.6276 (0.6276)	Top 1-err 18.3594 (18.3594)	Top 5-err 3.1250 (3.1250)
* Epoch: [99/100]	 Top 1-err 28.522  Top 5-err 9.358	 Test Loss 1.131
Current best accuracy (top-1 and 5 error): 28.428 9.406
Train with distillation: [Epoch 100/100][Batch 0/5005]	 Loss 2.131, Top 1-error 31.250, Top 5-error 14.062
Train with distillation: [Epoch 100/100][Batch 500/5005]	 Loss 2.069, Top 1-error 30.015, Top 5-error 11.888
Train with distillation: [Epoch 100/100][Batch 1000/5005]	 Loss 2.067, Top 1-error 30.093, Top 5-error 11.922
Train with distillation: [Epoch 100/100][Batch 1500/5005]	 Loss 2.072, Top 1-error 30.191, Top 5-error 12.020
Train with distillation: [Epoch 100/100][Batch 2000/5005]	 Loss 2.073, Top 1-error 30.238, Top 5-error 12.026
Train with distillation: [Epoch 100/100][Batch 2500/5005]	 Loss 2.073, Top 1-error 30.207, Top 5-error 12.030
Train with distillation: [Epoch 100/100][Batch 3000/5005]	 Loss 2.074, Top 1-error 30.215, Top 5-error 12.044
Train with distillation: [Epoch 100/100][Batch 3500/5005]	 Loss 2.073, Top 1-error 30.210, Top 5-error 12.038
Train with distillation: [Epoch 100/100][Batch 4000/5005]	 Loss 2.074, Top 1-error 30.223, Top 5-error 12.051
Train with distillation: [Epoch 100/100][Batch 4500/5005]	 Loss 2.074, Top 1-error 30.216, Top 5-error 12.039
Train with distillation: [Epoch 100/100][Batch 5000/5005]	 Loss 2.073, Top 1-error 30.216, Top 5-error 12.018
Train 	 Time Taken: 2514.83 sec
Test (on val set): [Epoch 100/100][Batch 0/196]	Time 1.744 (1.744)	Loss 0.6358 (0.6358)	Top 1-err 17.5781 (17.5781)	Top 5-err 3.5156 (3.5156)
* Epoch: [100/100]	 Top 1-err 28.482  Top 5-err 9.360	 Test Loss 1.132
Current best accuracy (top-1 and 5 error): 28.428 9.360
Best accuracy (top-1 and 5 error): 28.428 9.360