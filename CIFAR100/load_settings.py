import torch
import os

import models.WideResNet as WRN
import models.ResNet as RN


def load_paper_settings(args):

    WRN_40_2 = os.path.join(args.data_path, 'WRN40X2.pth')
    Resnet_56 = os.path.join(args.data_path, 'resnet_56.pth')
    Resnet_110 = os.path.join(args.data_path, 'resnet_110.pth')

    if args.paper_setting == 'a':
        teacher = RN.ResNet(depth=56, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_56, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_56, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=20, num_classes=100)

    elif args.paper_setting == 'b':
        teacher = RN.ResNet(depth=56, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_56, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_56, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=32, num_classes=100)

    elif args.paper_setting == 'c':
        teacher = RN.ResNet(depth=110, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_110, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_110, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=20, num_classes=100)

    elif args.paper_setting == 'd':
        teacher = RN.ResNet(depth=110, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_110, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_110, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=32, num_classes=100)

    elif args.paper_setting == 'e':
        teacher = RN.ResNet(depth=110, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_110, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_110, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=56, num_classes=100)

    elif args.paper_setting == 'f':
        teacher = RN.ResNet(depth=110, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_110, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_110, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=1, num_classes=100)

    elif args.paper_setting == 'g':
        teacher = RN.ResNet(depth=110, num_classes=100)
        if args.device == 0:
            state = torch.load(Resnet_110, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(Resnet_110, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'h':
        teacher = WRN.WideResNet(depth=40, widen_factor=2, num_classes=100)
        if args.device == 0:
            state = torch.load(WRN_40_2, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(WRN_40_2, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'i':
        teacher = WRN.WideResNet(depth=40, widen_factor=2, num_classes=100)
        if args.device == 0:
            state = torch.load(WRN_40_2, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(WRN_40_2, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=32, num_classes=100)

    elif args.paper_setting == 'j':
        teacher = WRN.WideResNet(depth=40, widen_factor=2, num_classes=100)
        if args.device == 0:
            state = torch.load(WRN_40_2, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(WRN_40_2, map_location={'cuda:1': 'cpu'})
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=56, num_classes=100)


    else:
        print('Undefined setting name !!!')

    return teacher, student, args