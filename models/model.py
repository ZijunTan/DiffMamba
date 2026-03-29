import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Mamba_backbone import Backbone_VSSM
from models.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from thop import profile
from models.decoder import DGMDM



class DEMFM(nn.Module):
    def __init__(self, in_d, out_d, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs):
        super(DEMFM, self).__init__()

        self.vss_cat = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_d * 2, out_channels=out_d),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_d, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'],
                     mlp_act_layer=mlp_act_layer,  mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity())

        self.vss_ver = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_d, out_channels=out_d),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_d, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'],
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity())

        self.vss_hor = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_d, out_channels=out_d),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_d, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'],
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity())

        self.vss_sub = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_d, out_channels=out_d),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_d, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'],
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity())


        self.con_en1 = nn.Sequential(nn.Conv2d(out_d * 2 + in_d, out_d, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_d),
                                      nn.ReLU(inplace=True))
        self.con_en2 = nn.Sequential(nn.Conv2d(out_d * 2 + in_d, out_d, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_d),
                                     nn.ReLU(inplace=True))

        self.conv_dr = nn.Sequential(nn.Conv2d(out_d * 3, out_d, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_d),
                                      nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_d),
                                      nn.ReLU(inplace=True))


    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x_cat = self.vss_cat(torch.cat([x1, x2], dim=1))

        x_hor = torch.stack([x1, x2], dim=-1)  # B,C,H,W,2
        x_hor = x_hor.reshape(b, c, h, 2 * w)
        x_hor = self.vss_hor(x_hor)
        x_hor1, x_hor2 = x_hor[:, :, :, ::2], x_hor[:, :, :, 1::2]

        x_ver = torch.stack([x1, x2], dim=-2)  # B,C,H,2,W
        x_ver = x_ver.reshape(b, c, 2 * h, w)
        x_ver = self.vss_ver(x_ver.transpose(-1, -2)).transpose(-1, -2)
        x_ver1, x_ver2 = x_ver[:, :, ::2, :], x_ver[:, :, 1::2, :]

        x_v1 = self.con_en1(torch.cat([x_hor1, x_ver1, x1], dim=1))
        x_v2 = self.con_en2(torch.cat([x_hor2, x_ver2, x2], dim=1))

        x = self.conv_dr(torch.cat([x_v1, x_v2, x_cat], dim=1))
        x_sub = self.vss_sub(torch.abs(x1 - x2))

        x = self.conv_out(x + x_sub)

        return x






class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()

        self.cls = nn.Conv2d(in_d, out_d, kernel_size=1, bias=True)

    def forward(self, d5, d4, d3, d2):

        # d5 = d5 + F.interpolate(d4, d5.shape[2:], mode='bilinear') + \
        #       F.interpolate(d3, d5.shape[2:], mode='bilinear') + F.interpolate(d2, d5.shape[2:], mode='bilinear')
        d5 = F.interpolate(d2, d5.shape[2:], mode='bilinear')
        mask = self.cls(d5)

        return mask


class BaseNet(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(BaseNet, self).__init__()
        output_nc = 1
        self.backbone = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        self.mid_d = kwargs['dims']
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.DEMFM5 = DEMFM(self.mid_d * 8, self.mid_d, norm_layer, self.backbone.channel_first, ssm_act_layer, mlp_act_layer, **clean_kwargs)
        self.DEMFM4 = DEMFM(self.mid_d * 4, self.mid_d, norm_layer, self.backbone.channel_first, ssm_act_layer, mlp_act_layer, **clean_kwargs)
        self.DEMFM3 = DEMFM(self.mid_d * 2, self.mid_d, norm_layer, self.backbone.channel_first, ssm_act_layer, mlp_act_layer, **clean_kwargs)
        self.DEMFM2 = DEMFM(self.mid_d, self.mid_d, norm_layer, self.backbone.channel_first, ssm_act_layer, mlp_act_layer, **clean_kwargs)


        self.DGMDM = DGMDM([self.mid_d, self.mid_d * 2, self.mid_d * 4, self.mid_d * 8],
                                     self.mid_d, norm_layer, self.backbone.channel_first, ssm_act_layer, mlp_act_layer, **clean_kwargs)


    def forward(self, x1, x2):

        x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        d5 = self.DEMFM5(x1_5, x2_5)  # 1/32
        d4 = self.DEMFM4(x1_4, x2_4)  # 1/16
        d3 = self.DEMFM3(x1_3, x2_3)  # 1/8
        d2 = self.DEMFM2(x1_2, x2_2)  # 1/4

        mask = self.DGMDM([x1_2, x1_3, x1_4, x1_5], [x2_2, x2_3, x2_4, x2_5], [d2, d3, d4, d5])
        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear')
        mask = torch.sigmoid(mask)

        return mask


if __name__ == '__main__':
    torch.cuda.set_device(7)
    path = '/home/students/doctor/2024/tanzj/PycharmProject/changedetection/A2Net/Config_VSSM/vssm_tiny_0230_ckpt_epoch_262.pth'
    model = BaseNet(pretrained=path, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2], dims=96,
                    ssm_d_state=1, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                    ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v0", forward_type='v3noz',
                    mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.2,
                    drop_rate=0.2, patch_norm=True, norm_layer="ln", downsample_version="v3",
                    patchembed_version="v2", gmlp=False, use_checkpoint=False).cuda()
    x = torch.randn((1, 3, 288, 288)).cuda()
    flops, params = profile(model, inputs=(x, x))
    print("parms=M", params / (1000 ** 2))
    print("flops=G", flops / (1000 ** 3))
    out = model(x, x)
    print(out.shape)


