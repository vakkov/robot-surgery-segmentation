import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi
from operator import itemgetter
from loss import class2one_hot

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import RasTerNetV2, TernausNetV2, UNet11, LinkNet34, UNet, UNet16, AlbuNet #DeeperNetV3
# from ternaus_v3_oc import TernausNetOC
from loss import LossBinary, LossMulti, FocalAndJaccardLoss, BCEAndLovaszLoss, LovaszSoftmax, SoftIoULoss, SurfaceLoss, Combined, Combined_Lovasz
from loss2 import BinaryDiceLoss, DiceLoss, Dice_loss, DICELoss
from modules.wasserstein import WassersteinDice
from dataset import RoboticsDataset
import utils
import sys
from prepare_train_val import get_split

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)

moddel_list = {#'TernausNetOC': TernausNetOC,
               #'DeeperNetV3': DeeperNetV3,
               'TernausNetV2': TernausNetV2,
               'RasTerNetV2': RasTerNetV2,
               'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--train_crop_height', type=int, default=1024)
    arg('--train_crop_width', type=int, default=1280)
    arg('--val_crop_height', type=int, default=1024)
    arg('--val_crop_width', type=int, default=1280)
    arg('--type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--model', type=str, default='UNet', choices=moddel_list.keys())

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    if not utils.check_crop_size(args.train_crop_height, args.train_crop_width):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.train_crop_height, train_crop_width=args.train_crop_width))
        sys.exit(0)

    if not utils.check_crop_size(args.val_crop_height, args.val_crop_width):
        print('Input image sizes should be divisible by 32, but validation '
              'crop sizes ({val_crop_height} and {val_crop_width}) '
              'are not.'.format(val_crop_height=args.val_crop_height, val_crop_width=args.val_crop_width))
        sys.exit(0)

    if args.type == 'parts':
        num_classes = 4
    elif args.type == 'instruments':
        num_classes = 8
    else:
        num_classes = 1

    if args.type == 'binary':
        #loss = LossBinary(jaccard_weight=args.jaccard_weight)
        #loss = FocalAndJaccardLoss(focal_weight=0.7, jaccard_weight=args.jaccard_weight, per_image=True)
        loss = BCEAndLovaszLoss(bce_weight=0.1, lovasz_weight=0.9, per_image=False)
    else:
        #loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)
        loss = LovaszSoftmax()
        
        #loss =  SoftIoULoss(n_classes=num_classes)
        
        #loss = SurfaceLoss(kwargs={"idc": [1,7], "num_classes": num_classes}, num_classes=num_classes)
        #loss = Combined(idc=[0, 1]) 
        #loss = Combined_Lovasz(idc=[0, 1])
        #loss = DiceLoss(n_classes=num_classes)
        
        # weights = (torch.ones((num_classes,1))).to(torch.device("cuda"))
        # loss = DICELoss(weights = weights)
        #loss = WassersteinDice(n_classes=num_classes)

    criterion_path = Path.joinpath(root, type(loss).__name__)
    criterion_path.mkdir(exist_ok=True, parents=True)
    
    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    # elif args.model == 'TernausNetOC':
    #     model = TernausNetOC(num_classes=num_classes, pretrained=True)    
    elif args.model == 'TernausNetV2':
        model = TernausNetV2(num_classes=num_classes, pretrained=True)
    elif args.model == 'RasTerNetV2':
        model = RasTerNetV2(num_classes=num_classes, pretrained=True)
    # elif args.model == 'DeeperNetV3':
    #     model = DeeperNetV3(num_classes=num_classes, pretrained=True)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)

    if torch.cuda.is_available():
        pin_memory=True
        torch.cuda.empty_cache()
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
            #pin_memory=True
        )

    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
            RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    def gt_transform(): 
        return Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=num_classes),
        itemgetter(0)])    

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type,
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type,
                               batch_size=len(device_ids))

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
