import timm
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

import segmentation_models_pytorch as smp


def replace_2d_by_1d(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## Compound module, go inside it
            replace_2d_by_1d(module)
        if isinstance(module, torch.nn.Sequential):
            ## Sequential module, go inside it
            for submodule in module:
                replace_2d_by_1d(module)

        if isinstance(module, torch.nn.Conv2d):
            ## Conv2d -> Conv1d:
            padding = module.padding[0]
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]

            module = nn.Conv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=module.groups,
                padding=padding,
                bias=(torch.is_tensor(module.bias) or (module.bias == True)),
                dilation=1,
            )
            setattr(model, n, module)

        if isinstance(module, nn.BatchNorm2d):
            module = nn.BatchNorm1d(
                module.num_features,
                track_running_stats=module.track_running_stats,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
            )
            setattr(model, n, module)

        if isinstance(module, timm.layers.SelectAdaptivePool2d):
            module = nn.AdaptiveAvgPool1d(
                output_size=module.pool.output_size,
            )
            setattr(model, n, module)

        if isinstance(module, nn.MaxPool2d):
            module = nn.MaxPool1d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                return_indices=module.return_indices,
                ceil_mode=module.ceil_mode,
            )
            setattr(model, n, module)

## CONV-BASED MODELS ___________________________________________________________________________________________________
# ENCODER:
class Encoder1d(nn.Module):
    def __init__(
        self,
        timm_encoder_name="resnet18",
        latent_dim=32,
        converter_2dto1d=replace_2d_by_1d,
        num_featuremaps=512,
        position_encoding=False
    ):
        super().__init__()
        
        timm_model = timm.create_model(
            timm_encoder_name, features_only=True, in_chans=1
        )
        converter_2dto1d(timm_model)
        self.features_extractor = timm_model
        self.position_encoding = position_encoding

        if self.position_encoding:
            num_featuremaps *= 2
        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=num_featuremaps, out_channels=latent_dim, kernel_size=(1,)
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
            
    def forward(self, x):
        features = self.features_extractor(x)[-1]

        if self.position_encoding:
            skull_pos = self.find_skull_pos(x, features)
            x = torch.cat((features, skull_pos), dim=1)
        else:
            x = features
        x = self.projection(x)
        return x

    def find_skull_pos(self, x, features):
        skull_pos = ((x > 0.15).float().argmax(dim=-1).unsqueeze(-1) * torch.ones_like(features))
        return skull_pos

# DECODER:
class Conv1dAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3,),
                padding=(1,),
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv1 = Conv1dAct(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = Conv1dAct(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Decoder1d(nn.Module):
    def __init__(self, latent_dim=32, target_len=256):
        super().__init__()

        # TODO: пнужно переделать проверку, сичтать логарифмы
        if target_len % latent_dim:
            raise RuntimeError("Invalid target_len")
        numblocks = (target_len // latent_dim) // 2
        # ...

        in_channels = [4**i for i in range(5 - numblocks, 5)[::-1]]
        out_channels = in_channels[1:] + [1]
        scale = [4] * numblocks

        self.projection = nn.Linear(in_features=latent_dim, out_features=in_channels[0])
        blocks = [
            UpBlock1d(_in, _out, _scale)
            for _in, _out, _scale in zip(in_channels, out_channels, scale)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x =  self.projection(x)
        x = x.view(-1, self.projection.out_features, 1)
        x = self.blocks(x)
        return x

# AUTO-ENCODER:
class AE_model(nn.Module):
    def __init__(self, latent_dim=32, target_len=256, num_featuremaps=512, timm_encoder_name='resnet18', position_encoding=False): # timm_encoder_name='efficientnet_lite2', num_featuremaps=352
        super().__init__()
        target_len = target_len * int(latent_dim / 32)
        self.encoder = Encoder1d(latent_dim=latent_dim,
                                 timm_encoder_name=timm_encoder_name,
                                 num_featuremaps=num_featuremaps,
                                 position_encoding=position_encoding)
        self.decoder = Decoder1d(latent_dim=latent_dim,
                                 target_len=target_len)

    def forward(self, x):
        return self.decoder(self.encoder(x))

## LINEAR MODELS _______________________________________________________________________________________________________
# ENCODER:
class EncoderLinear(nn.Module):
    def __init__(
        self, 
        in_features=256,
        latent_dim=32, 
        num_featuremaps=512
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=num_featuremaps),
            nn.BatchNorm1d(1),
            nn.GELU(),
            nn.Linear(in_features=num_featuremaps, out_features=num_featuremaps),
            nn.BatchNorm1d(1),
            nn.GELU(),
            nn.Linear(in_features=num_featuremaps, out_features=latent_dim),
            nn.BatchNorm1d(1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x).squeeze()

# DECODER:
class DecoderLinear(nn.Module):
    def __init__(
        self,
        latent_dim=32, 
        num_featuremaps=512,
        out_features=256
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=num_featuremaps),
            nn.BatchNorm1d(1),
            nn.GELU(),
            nn.Linear(in_features=num_featuremaps, out_features=num_featuremaps),
            nn.BatchNorm1d(1),
            nn.GELU(),
            nn.Linear(in_features=num_featuremaps, out_features=out_features),
            nn.BatchNorm1d(1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x.unsqueeze(1))

# AUTO-ENCODER:
class AE_linear(nn.Module):
    def __init__(
        self,
        series_len=256,
        latent_dim=32,
        num_featuremaps=512
    ):
        super().__init__()
        self.encoder = EncoderLinear(
            in_features=series_len,
            latent_dim=latent_dim, 
            num_featuremaps=num_featuremaps
        )
        self.decoder = DecoderLinear(
            latent_dim=latent_dim, 
            num_featuremaps=num_featuremaps,
            out_features=series_len
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

## CROP-BASED MODELS ___________________________________________________________________________________________________
# ENCODER:
class EncoderWCrop1d(nn.Module):
    def __init__(
        self,
        in_features=256,
        latent_dim=32,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=self.latent_dim, kernel_size=(1,), bias=False
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
            
    def forward(self, x):
        x = x.view(-1,1,x.shape[-1])
        threshold = 0.15 
        
        skull_pos = (x > threshold).float().argmax(dim=-1, keepdim=True)
        x_cut = torch.ones((x.shape[0], 1, 2*self.latent_dim)).to(x.device) * skull_pos / 256
        x = torch.cat((x, x[:,:,:]), dim=2)
        x_cut[:, 0, :self.latent_dim]= x[
            torch.arange(x.shape[0]).unsqueeze(1),
            0,
            skull_pos.squeeze(2) + torch.arange(32).to(x.device)
        ]
        # self.projection = self.projection.to(x.device)
        # features = self.projection(x_cut)
        return x_cut #features

# AUTO-ENCODER:
class AE_cropper(nn.Module):
    def __init__(
        self,
        series_len=256,
        latent_dim=32,
        num_featuremaps=512
    ):
        super().__init__()
        self.encoder = EncoderWCrop1d(
            in_features=series_len,
            latent_dim=latent_dim,
        )
        self.decoder = DecoderLinear(
            latent_dim=latent_dim,
            num_featuremaps=num_featuremaps,
            out_features=series_len
        )

    def forward(self, x):
        x = self.encoder(x)
        self.decoder = self.decoder.to(x.device)
        return self.decoder(x)

## CROP-BASED MODELS _______________________________________________________________________________________________________