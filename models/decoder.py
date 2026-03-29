import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute



class Decoder_Block(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs):
        super(Decoder_Block, self).__init__()

        self.conv_in1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_dim),
                                      nn.ReLU(inplace=True))

        self.conv_in2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_dim),
                                      nn.ReLU(inplace=True))

        self.st_block1 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_dim * 3, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block2 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block3 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block4 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block5 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=out_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.conv_fuse1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=out_dim * 5, out_channels=out_dim, bias=False),
                                        nn.BatchNorm2d(out_dim),
                                        nn.ReLU(inplace=True))
        self.conv_fuse2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=out_dim * 5, out_channels=out_dim, bias=False),
                                        nn.BatchNorm2d(out_dim),
                                        nn.ReLU(inplace=True))
        self.conv_fuses = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=out_dim * 5, out_channels=out_dim, bias=False),
                                        nn.BatchNorm2d(out_dim),
                                        nn.ReLU(inplace=True))


        self.conv_sub = nn.Sequential(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_dim),
                                      nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_dim),
                                      nn.ReLU(inplace=True))

    def forward(self, x1, x2, x_sub):

        x1 = self.conv_in1(x1)
        x2 = self.conv_in2(x2)

        B, C, H, W = x_sub.size()
        p1 = self.st_block1(torch.cat([x_sub, x1, x2], dim=1))
        p1s, p11, p12 = p1.chunk(3, dim=1)

        ct_tensor_2 = torch.stack([x_sub, x1, x2], dim=-1)  # B,C,H,W,3
        ct_tensor_2 = ct_tensor_2.reshape(B, C, H, 3 * W)
        p2 = self.st_block2(ct_tensor_2).reshape(B, C, H, W, 3)
        p2s, p21, p22 = p2[:, :, :, :, 0], p2[:, :, :, :, 1], p2[:, :, :, :, 2]

        ct_tensor_3 = torch.stack([x_sub, x1, x2], dim=-2)  # B,C,H,W,3
        ct_tensor_3 = ct_tensor_3.reshape(B, C, 3 * H, W)
        p3 = self.st_block3(ct_tensor_3.transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, 3, W)
        p3s, p31, p32 = p3[:, :, :, 0, :], p3[:, :, :, 1, :], p3[:, :, :, 2, :]

        ct_tensor_4 = torch.cat([x_sub, x1, x2], dim=-1)
        p4 = self.st_block4(ct_tensor_4)
        p4s, p41, p42 = p4.chunk(3, dim=-1)

        ct_tensor_5 = torch.cat([x_sub, x1, x2], dim=-2)
        p5 = self.st_block5(ct_tensor_5.transpose(-1, -2)).transpose(-1, -2)
        p5s, p51, p52 = p5.chunk(3, dim=-2)

        p_1 = self.conv_fuse1(torch.cat([p11, p21, p31, p41, p51], dim=1))
        p_2 = self.conv_fuse2(torch.cat([p12, p22, p32, p42, p52], dim=1))
        p_s = self.conv_fuses(torch.cat([p1s, p2s, p3s, p4s, p5s], dim=1))

        p = self.conv_sub(torch.abs(p_1-p_2))
        p = self.conv_out(p + p_s)

        return p




class DGMDM(nn.Module):
    def __init__(self, encoder_dims, mid_dim, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs):
        super(DGMDM, self).__init__()

        self.decoder_4 = Decoder_Block(encoder_dims[-1], mid_dim, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs)
        self.decoder_3 = Decoder_Block(encoder_dims[-2], mid_dim, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs)
        self.decoder_2 = Decoder_Block(encoder_dims[-3], mid_dim, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs)
        self.decoder_1 = Decoder_Block(encoder_dims[-4], mid_dim, norm_layer, channel_first, ssm_act_layer, mlp_act_layer, **kwargs)

        self.smooth_layer_3 = nn.Sequential(nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(mid_dim),
                                            nn.ReLU(inplace=True))
        self.smooth_layer_2 = nn.Sequential(nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(mid_dim),
                                            nn.ReLU(inplace=True))
        self.smooth_layer_1 = nn.Sequential(nn.Conv2d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(mid_dim),
                                            nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(in_channels=mid_dim, out_channels=1, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features, sub_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features
        d1, d2, d3, d4 = sub_features

        p4 = self.decoder_4(pre_feat_4, post_feat_4, d4)

        p3 = self.decoder_3(pre_feat_3, post_feat_3, d3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        p2 = self.decoder_2(pre_feat_2, post_feat_2, d2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        p1 = self.decoder_1(pre_feat_1, post_feat_1, d1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        mask = self.cls(p1)

        return mask


