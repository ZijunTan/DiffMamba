import os
import sys
from models.model import BaseNet
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
import time
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import logging
import datetime
from config import get_config


torch.set_num_threads(3)


def init_logging(filedir: str):
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger()
    fh = logging.FileHandler(filename=filedir + '/log_' + get_date_str() + '.txt')
    sh = logging.StreamHandler()
    formatter_fh = logging.Formatter('%(asctime)s %(message)s')
    formatter_sh = logging.Formatter('%(message)s')
    fh.setFormatter(formatter_fh)
    sh.setFormatter(formatter_sh)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    fh.setLevel(10)
    sh.setLevel(10)
    return logging


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def BCE(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    return bce


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)

    for iter, batched_inputs in enumerate(tqdm(val_loader)):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()

        pre_img = pre_img.cuda()
        target = target.cuda()
        post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

    scores = salEvalVal.get_scores()

    return scores


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()
    epoch_loss = []

    tar = tqdm(train_loader, ncols=120)
    for iter, batched_inputs in enumerate(tar):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        pre_img = pre_img.cuda()
        target = target.cuda()
        post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output = model(pre_img_var, post_img_var)
        loss = BCEDiceLoss(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())

        tar.set_description('iteration: [%d/%d] lr: %.7f loss: %.4f' %
                            (iter + cur_iter, max_batches * args.max_epochs, lr, loss.data.item()))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def trainValidateSegmentation(args, config):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = BaseNet(pretrained=args.pretrained_weight_path,
                    patch_size=config.MODEL.VSSM.PATCH_SIZE,
                    in_chans=config.MODEL.VSSM.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    depths=config.MODEL.VSSM.DEPTHS,
                    dims=config.MODEL.VSSM.EMBED_DIM,
                    # ===================
                    ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                    ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                    ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                    ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                    ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                    ssm_conv=config.MODEL.VSSM.SSM_CONV,
                    ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                    ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                    ssm_init=config.MODEL.VSSM.SSM_INIT,
                    forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                    # ===================
                    mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                    mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                    mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                    # ===================
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    drop_rate=args.drop_rate,
                    patch_norm=config.MODEL.VSSM.PATCH_NORM,
                    norm_layer=config.MODEL.VSSM.NORM_LAYER,
                    downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                    patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                    gmlp=config.MODEL.VSSM.GMLP,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    args.savedir = args.savedir + '/' + args.name
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    logging = init_logging(args.savedir)

    if args.file_root == 'LEVIR':
        args.file_root = '/home/students/doctor/2024/tanzj/Dataset/CD/LEVIR'
    elif args.file_root == 'Fire':
        args.file_root = '/home/students/doctor/2024/tanzj/Dataset/CD/Fire'
    else:
        raise TypeError('%s has not defined' % args.file_root)


    model = model.cuda()

    logging.info(f"args: {args}\t")


    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 256. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root, transform=trainDataset_main)
    trainLoader = torch.utils.data.DataLoader(train_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=False,
                                              drop_last=False)

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(test_data,
                                             shuffle=False,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             pin_memory=False)


    max_batches = len(trainLoader)

    logging.info('For each epoch, we have {} batches'.format(max_batches))

    cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0

    if args.resume is not None:
        args.resume = args.savedir + '/checkpoint.pth.tar'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * len(trainLoader)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    best_epoch = 0
    best_F1 = 0.0

    for epoch in range(start_epoch, args.max_epochs):
        logging.info('\nEpoch {}/{}, Best F1 in epoch {}, Best F1 = {:.2f}'.format(epoch+1, args.max_epochs, best_epoch, best_F1 * 100))
        lossTr, lr = train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)
        logging.info('loss = {:.4f}'.format(lossTr))
        cur_iter += len(trainLoader)
        torch.cuda.empty_cache()

        score_val = val(args, testLoader, model)
        torch.cuda.empty_cache()

        logging.info('Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, IoU: {:.2f}'.format(
            score_val['precision'] * 100, score_val['recall'] * 100, score_val['F1'] * 100,  score_val['IoU'] * 100))

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + '/' + 'checkpoint.pth.tar')

        # save the model also
        model_file_name = args.savedir + '/' + 'best_model.pth'
        if best_F1 <= score_val['F1']:
            best_F1 = score_val['F1']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)

        torch.cuda.empty_cache()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', default="DiffMamba")
    parser.add_argument('--file_root', default="Fire", help='| LEVIR | Fire ')
    parser.add_argument('--inWidth', type=int, default=288, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=288, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations 5000')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size  32')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1.25e-4, help='Initial learning rate 5e-4')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./Checkpoint', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training | '
                                                       './results_ep100/checkpoint.pth.tar')
    ######
    parser.add_argument('--cfg', type=str, default='/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Config_VSSM/vssm_tiny_224.yaml')
    parser.add_argument('--pretrained_weight_path', type=str, default='/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Config_VSSM/vssm_tiny_0230_ckpt_epoch_262.pth')
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    ######

    args = parser.parse_args()
    config = get_config(args)

    trainValidateSegmentation(args, config)