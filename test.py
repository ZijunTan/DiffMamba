import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from models.model import BaseNet
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from config import get_config



@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    for iter, batched_inputs in enumerate(tqdm(val_loader)):

        img, target = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        pre_img = pre_img.cuda()
        target = target.cuda()
        post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # save change maps
        pr = pred[0, 0].cpu().numpy()
        gt = target_var[0, 0].cpu().numpy()

        if args.save is True:
            if not os.path.exists(args.vis_dir):
                os.makedirs(args.vis_dir)
            index_tp = np.where(np.logical_and(pr == 1, gt == 1))
            index_fp = np.where(np.logical_and(pr == 1, gt == 0))
            index_tn = np.where(np.logical_and(pr == 0, gt == 0))
            index_fn = np.where(np.logical_and(pr == 0, gt == 1))
            #
            map = np.zeros([gt.shape[0], gt.shape[1], 3])
            map[index_tp] = [255, 255, 255]  # white
            map[index_fp] = [255, 0, 0]  # red
            map[index_tn] = [0, 0, 0]  # black
            map[index_fn] = [0, 255, 0]  # Cyan

            change_map = Image.fromarray(np.array(map, dtype=np.uint8))
            change_map.save(args.vis_dir + img_name)

        f1 = salEvalVal.update_cm(pr, gt)

    scores = salEvalVal.get_scores()

    return scores


def ValidateSegmentation(args, config):
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

    if args.file_root == 'Fire':
        args.file_root = '/home/students/doctor/2024/tanzj/Dataset/CD/Fire'
    elif args.file_root == 'BCDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/SYSU'
    elif args.file_root == 'CDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/CDD'
    elif args.file_root == 'Fire':
        args.file_root = '/home/students/doctor/2024/tanzj/Dataset/CD/Fire'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(test_data, shuffle=False,  batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=False)

    cudnn.benchmark = True

    state_dict = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    model = model.cuda()

    score = val(args, testLoader, model)
    print('Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}, IoU: {:.2f}'.format(
        score['precision'] * 100, score['recall'] * 100, score['F1'] * 100, score['IoU'] * 100))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="Fire", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=288, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=288, help='Height of RGB image')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--weight_path', default='/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Checkpoint/test/best_model.pth')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--vis_dir', default='/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Result/')

    ######
    parser.add_argument('--cfg', type=str, default='/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Config_VSSM/vssm_tiny_224.yaml')
    parser.add_argument('--pretrained_weight_path', type=str, default='/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Config_VSSM/vssm_tiny_0230_ckpt_epoch_262.pth')
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    ######

    args = parser.parse_args()
    config = get_config(args)
    print('Called with args:')
    print(args)

    ValidateSegmentation(args, config)
