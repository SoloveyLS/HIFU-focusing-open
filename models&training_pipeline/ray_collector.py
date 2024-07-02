import torch
import torch.nn as nn
from embedder import EncoderWCrop1d

#  input of collector should be:    n_skulls*256**2 x 32
# output of collector should be:    n_skulls        x 32 x 256 x 256:
class Ray_collector(nn.Module):
    def __init__(
        self, 
        embedder=EncoderWCrop1d(),
    ):
        super().__init__()
        self.embedder = embedder
    
    def forward(self, x):
        x = self.embedder(x)
        if x.shape[0] % 256**2 != 0:
            raise RuntimeError('Wrong number of rays came into collector!')
        n_skulls = int(x.shape[0] / 256**2)
        
        skull_data = torch.zeros((n_skulls, 64, 256, 256)).to(x.device)
        for ind in range(n_skulls):
            skull_data[ind, :, :, :] = self.collect(x[(256**2 * ind) : (256**2 * (ind+1)), :])
        
        return skull_data
        
    def collect(self, x):
        # x.shape = (256**2,32); output.shape = (1,32,256,256):
        x = x.squeeze()
        if x.shape != torch.Size([256**2, 64]):
            print(x.shape)
            raise RuntimeError('Inner Ray_collector error: collector obtained wrong tensor.Size')
        return x.permute((1,0)).unsqueeze(0).reshape((1,64,256,256))

class Ray_collector_pass(nn.Module):
    def __init__(
        self, 
        embedder=EncoderWCrop1d(),
    ):
        super().__init__()
        self.embedder = embedder
    
    def forward(self, x):
        return x