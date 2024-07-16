'''3D model for distillation.'''

from collections import OrderedDict
from .minkunet import mink_unet as model3D
from .resunet import ResUNetBN2C as FCGF3D
from torch import nn


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    return model

def FCGF3d(**kwargs):
    model = model3D(**kwargs)
    return model

class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, cfg=None):
        super(DisNet, self).__init__()
        if not hasattr(cfg, 'feature_2d_extractor'):
            cfg.feature_2d_extractor = 'openseg'
        if 'lseg' in cfg.feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in cfg.feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # MinkowskiNet for 3D point clouds
        if cfg.fcgf:
            net3d = FCGF3d(in_channels=1, out_channels=last_dim, D=3)
        else:
            net3d = constructor3d(in_channels=3, out_channels=last_dim, D=3, arch=cfg.arch_3d)
        self.net3d = net3d
    
    def forward(self, sparse_3d):
        '''Forward method.'''

        out_field = self.net3d(sparse_3d)

        out_features_0 = (out_field.slice(sparse_3d)).F

        return out_features_0
