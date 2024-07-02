import torch
import torch.nn as nn
import timm

## depth conv3d:
class dsConv3d(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, kernel_size=3, stride=1, padding=1, bias = True):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride = stride, 
                                   padding = padding, bias= bias, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x, weight = None):
        if weight is not None:
            self.depthwise.weight = torch.nn.Parameter(weight)
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# 3d version of timm BatchNormAct2d
class BatchNormAct3d(nn.BatchNorm3d): # just a copypaste of BNA2 from hugginface (2d -> 3d, not sure if working)
    """BatchNorm + Activation
    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct3d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.act = act_layer
        self.drop = nn.Identity()

    def _forward_jit(self, x):
        """ A cut & paste of the contents of the PyTorch BatchNorm3d forward function
        """
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        x = F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        return x

    @torch.jit.ignore
    def _forward_python(self, x):
        return super(BatchNormAct3d, self).forward(x)

    def forward(self, x):
        # FIXME cannot call parent forward() and maintain jit.script compatibility?
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.drop(x)
        x = self.act (x)
        return x


## 2d replacer
def replace_2d_by_3d(model, is_group_norm=False):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## Compound module, go inside it
            replace_2d_by_3d(module)
        if isinstance(module, torch.nn.Sequential):
            ## Sequential module, go inside it
            for submodule in module:
                replace_2d_by_3d(module)
        
        if isinstance(module, torch.nn.Conv2d):
            ## Conv2d -> Conv3d:
            padding     = (module.padding[0],     module.padding[0],     module.padding[0])
            kernel_size = (module.kernel_size[0], module.kernel_size[0], module.kernel_size[0])
            stride      = (module.stride[0],      module.stride[0],      module.stride[0])
    
            module = nn.Conv3d(in_channels =module.in_channels,
                               out_channels=module.out_channels,
                               kernel_size =kernel_size,
                               stride      =stride,
                               groups      =module.groups,
                               padding     =padding,
                               bias        =(torch.is_tensor(module.bias) or (module.bias == True)),
                               dilation    =(1,1,1)
                              )
            setattr(model, n, module)
            
        if isinstance(module, nn.BatchNorm2d):
            ## BatchNormAct2d -> BatchNormAct3d:
            module = BatchNormAct3d(num_features       =module.num_features, 
                                    eps                =module.eps,
                                    momentum           =module.momentum,
                                    affine             =module.affine,
                                    track_running_stats=module.track_running_stats,
                                    act_layer          =module.act,
                                    drop_block         =module.drop
                                   )
            setattr(model, n, module)
            
        if isinstance(module, timm.layers.SelectAdaptivePool2d):
            ## SelectAdaptivePool2d -> AdaptivePool3d:
            print(module.pool.output_size)
            module = nn.AdaptiveAvgPool3d(output_size=module.pool.output_size,
                                      )
            setattr(model, n, module)
        
        if isinstance(module, BatchNormAct3d) and is_group_norm:
            ## Well, we've changed BNA2d -> BNA3d, let's make it GN :D
            module = nn.Sequential(nn.GroupNorm(1, module.num_features), 
                                   nn.ReLU(),
                                   nn.Identity()
                                  )
            setattr(model, n, module)


## main func for model creation
def create_model_phase_prediction():
    model = timm.create_model('tf_efficientnet_b2')
    replace_2d_by_3d(model)
    model.classifier = nn.Sequential(nn.Flatten(),
                                     nn.Linear(1408, 256))
    model.conv_stem = nn.Conv3d(2, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), bias=False)
    # print(model)
    return model

def create_model_phase_prediction_b0():
    model = timm.create_model('tf_efficientnet_b0')
    replace_2d_by_3d(model)
    model.classifier = nn.Sequential(nn.Flatten(),
                                     nn.Linear(1280, 256))
    model.conv_stem = nn.Conv3d(2, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), bias=False)
    # print(model)
    return model