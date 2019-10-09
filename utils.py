import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
import tqdm
import sys
from tensorboardX import SummaryWriter

def cuda(x):
    if sys.version_info <= (3, 6):
        return x.cuda(async=True) if torch.cuda.is_available() else x
#    else:
#        return x.cuda(non_blocking=True) if torch.cuda.is_available() else x

def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2    


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    criterion_name = type(criterion).__name__ + "_" + args.model
    root = Path(args.root)
    model_path = root / criterion_name/args.type/ 'model_{fold}.pt'.format(fold=fold)
    writer = SummaryWriter(root/criterion_name/args.type/ 'model_{fold}'.format(fold=fold)) 
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    log = root.joinpath(criterion_name).joinpath(args.type).joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
            #for i, (inputs, targets, mask_onehot, mask_distmap) in enumerate(tl):
                inputs = cuda(inputs)
                #mask_onehot = cuda(mask_onehot)
                #mask_distmap = cuda(mask_distmap)

                with torch.no_grad():
                    targets = cuda(targets)

                r = np.random.rand(1)
                
                if args.augment and args.beta > 0 and r < args.cutmix_prob:
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(inputs.size()[0]).cuda()
                    #target_a = targets
                    target_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    # compute output
                    input_var = torch.autograd.Variable(inputs, requires_grad=True)
                    target_a_var = torch.autograd.Variable(targets)
                    target_b_var = torch.autograd.Variable(target_b)
                    outputs = model(input_var)
                    loss = criterion(outputs, target_a_var) * lam + criterion(outputs, target_b_var) * (1. - lam)

                else:
                     outputs = model(inputs)
                     loss = criterion(outputs, targets)
                #loss = criterion(outputs, targets, mask_onehot, mask_distmap)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                #loss.backward()
                loss.mean().backward()
                optimizer.step()
                step += 1
                #step = step + 1
                tq.update(batch_size)
                #print(loss.mean())
                #losses.append(loss.mean().item())
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    #writer.add_scalar(criterion_name/args.type + '/mean_loss', mean_loss, step)
                    writer.add_scalar(criterion_name + '/mean_loss', mean_loss, step)
                #torch.cuda.empty_cache()
            write_event(log, step, loss=mean_loss)
            #writer.add_scalar(criterion_name/args.type + '/epoch_mean_loss', mean_loss, step)
            writer.add_scalar(criterion_name + '/epoch_mean_loss', mean_loss, step)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            #writer.add_histogram(criterion_name + '/histogram', np.array(valid_metrics), step)
            valid_loss = valid_metrics['valid_loss']
            if args.type == 'binary':
                jaccard_loss = valid_metrics['jaccard_loss']
            else:
                average_iou = valid_metrics['iou']
                average_dice = valid_metrics['avg_dice']
            valid_losses.append(valid_loss)
            #writer.add_scalar(criterion_name/args.type + '/valid_loss', valid_loss, step)
            writer.add_scalar(criterion_name + '/valid_loss', valid_loss, step)
            if args.type == 'binary':
                #writer.add_scalar(criterion_name/args.type + '/jaccard_valid_loss', jaccard_loss, step)
                writer.add_scalar(criterion_name + '/jaccard_valid_loss', jaccard_loss, step)
            else:
                # writer.add_scalar(criterion_name/args.type + '/avg_iou', average_iou, step)
                # writer.add_scalar(criterion_name/args.type + '/avg_dice', average_dice, step)
                writer.add_scalar(criterion_name + '/avg_iou', average_iou, step)
                writer.add_scalar(criterion_name + '/avg_dice', average_dice, step)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return