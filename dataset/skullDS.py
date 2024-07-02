import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import hdf5storage
import scipy.io
import time

class FocusObj():
    def __init__(self, focus, type, additional_name):
        # raise RuntimeError('IDK...')
        self.additional_name = additional_name
        if type:
            self.phase =list(focus['output'][0][0].reshape(256))
            self.hologram = torch.tensor(focus['output'][0][1])
            self.transpose = focus['output'][0][2].reshape(3)
            self.skull_num = int(focus['output'][0][3].reshape(1))
        else:
            self.phase     = focus['output']['phase'][0][0][0]
            self.transpose = focus['output']['transpose'][0][0][0]
            self.skull_num = int(focus['output']["skull_num"][0][0][0][0])
        self.phase     = torch.tensor(self.phase    )
        self.transpose = torch.tensor(self.transpose)
    
    def get(self):
        return self.phase, self.transpose, self.skull_num, self.additional_name

# _________________________________________________________________________________________________________________
# DATASET: (x,y) = (shifted skull data, phases)
class SkullPhaseDS(Dataset):
    max_c0    = 3500
    water_c0  = 1500
    delta = 3
    
    def __init__(self, folder, max_buf_len=100, valid=False, focal_channel=False):
    # folder should be with '/' slashes
        super().__init__()
        self.folder = folder
        print(self.folder)
        self.skulls = {}
        self.focuses = []
        self.focus_type = []
        self.max_buf_len = max_buf_len
        self.focal_channel = focal_channel
        self.len = 0

        self.valid = valid
        
        self.time_per_file = 0
        self.__count_len()

    def __count_len(self):
    # function to count length of dataset:
        self.len = 0
        if self.valid:
            additional_name = '_valid'
        else:
            additional_name = ''
        # start timer
        start_time = time.time()
        while True:
            try:
            # we assume there is no skips:
                try:
                    buf = scipy.io.loadmat(self.folder + "/focuses" + f"/{(self.len+1):d}" + additional_name +".mat")
                    self.focuses.append(FocusObj(buf, 0, additional_name))
                    self.len += 1
                except:
                    buf = hdf5storage.loadmat(self.folder + "/focuses" + f"/{(self.len+1):d}" + additional_name + ".mat")
                    self.focuses.append(FocusObj(buf, 1, additional_name))
                    self.len += 1
            except:
                break
        # end timer
        end_time = time.time()
    
    def _read_file(self, ind):
    # function to read file by index
        # read focal point struct and extract skull number from it:
        if ind > self.len:
            raise KeyError("We've got less data then we want to! :c")
        # if self.valid:
        #     additional_name = '_valid'
        # else:
        #     additional_name = ''

        # focus_ = self.focuses[ind]['output']
        phase, transpose, skull_num, additional_name = self.focuses[ind].get()
        
        # if self.focus_type[ind]:
        #     phase =list(focus_['phase'][0].reshape(256))
        #     transpose = focus_['focus'][0][0]
        #     skull_num = int(focus_['skull_num'][0][0][0])
        # else:
        #     phase     = focus_['phase'][0][0][0]
        #     transpose = focus_['transpose'][0][0][0]
        #     skull_num = int(focus_["skull_num"][0][0][0][0])
            
        # try:
        #     focus_ = scipy.io.loadmat(self.folder + "/focuses" + f"/{ind:d}" + additional_name + ".mat")
        #     focus_ = focus_['output']
    
        #     phase     = focus_['phase'][0][0][0]
        #     transpose = focus_['transpose'][0][0][0]
        #     skull_num = int(focus_["skull_num"][0][0][0][0])
        # except:
        #     focus_ = hdf5storage.loadmat(self.folder + "/focuses" + f"/{ind:d}" + additional_name + ".mat")
        #     focus_ = focus_['output']
            
        #     phase =list(focus_['phase'][0].reshape(256))
        #     transpose = focus_['focus'][0][0]
        #     skull_num = int(focus_['skull_num'][0][0][0])
        
        focus_ = {}
        focus_['phases'] = torch.tensor(phase)
        focus_['skull_num'] = skull_num
        focus_['transpose'] = torch.tensor(transpose)
        del phase, transpose
        # focus_ = self.__normilize_phase(focus_)
        
        if skull_num not in self.skulls.keys():
        # if this skull not in buffer, making sure that buffer won't be more then self.max_buf_len after pushing new skull: 
            while len(self.skulls) >= self.max_buf_len:
                self.skulls.pop(min(self.skulls.keys()))
                
        # and just reading new skull and pushing into the buffer:
            try:
                skull_ = scipy.io.loadmat(self.folder + "/skulls" + f"/{skull_num:d}" + additional_name + ".mat")
                skull_ = torch.tensor(skull_['MaterialMatrix']['c0'][0][0])
            except:
                skull_ = hdf5storage.loadmat(self.folder + "/skulls" + f"/{skull_num:d}" + additional_name + ".mat")
                skull_ = torch.tensor(skull_['MaterialMatrix']['c0'][0])
            
            skull_ = self.__normilize_skull(skull_)
            self.skulls[skull_num] = skull_
        
        skull_ = self.skulls[skull_num]
        return skull_, focus_

    def __normilize_skull(self, skull_):
    # function to normilize skull data to [-1,1]
        # skull -> (skull - <water>) / (<max(skull)> - <water>)
        skull_ -= self.water_c0
        skull_ /= (self.max_c0-self.water_c0)

        return skull_
        
    def __len__(self):
    # ordinary len func
        return self.len
        
    def __getitem__(self, ind):
    # ordinary getter
        skull_, focus_ = self._read_file(ind+1)
        if self.focal_channel:
            skull_ = self.add_focal_channel(skull_, focus_['transpose'])#.unsqueeze(0)
        else:
            skull_ = skull_.unsqueeze(0)
        phases_ = focus_['phases']
        return skull_, phases_

    def __plotting_getter(self, ind):
    # ordinary getter
        skull_, focus_ = self._read_file(ind+1)
        skull_ = self.add_focal_channel(skull_, focus_['transpose'])#.unsqueeze(0)
        phases_ = focus_['phases']
# delete last one
        return skull_, phases_, focus_['transpose']

    def plot(self, ind, figsize=10):
        transform = T.ToPILImage()
        if type(ind) == int:
            x, _, transpose = self.__plotting_getter(ind)
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            ax.imshow(transform((x[140, :, :]+1)/2))
            ax.plot(180-transpose[2], 140-transpose[1], 'r+')
        else:
            fig, axs = plt.subplots(len(ind), 1, figsize=(figsize, len(ind)*figsize))
            for i, it in enumerate(ind):
                axs[i].set_title(f'Point number {it:d}:')
# delete last one
                x, _, transpose = self.__plotting_getter(it)
                x = x.squeeze()
                axs[i].imshow(transform((x[1, 140-transpose[0], :, :]+1))) 
                #  + 0.1*(self.transpose_skull(x[0,:,:,:]+1, transpose, water_val=1)/4)[140-transpose[0], :, :]
        
    @staticmethod
    def transpose_skull(skull_, transpose, water_val=0):
    # static method for skull data transposal
        skull_ = skull_.roll(tuple(transpose.to(torch.int16)), dims=[0,1,2])
        skull_[:, :, :transpose.to(torch.int16)[2]] = water_val
        return skull_
    @staticmethod
    def add_focal_channel(skull_, transpose, water_val=0):
        focal_channel = torch.zeros_like(skull_)

        x0 = 140-transpose[0]
        y0 = 140-transpose[1]
        z0 = 180-transpose[2]
        delta = SkullPhaseDS.delta
        
        inR = lambda x,y,z: ((x**2 + y**2 + z**2) <= delta**2)
        for itx in range(-delta, delta + 1):
            for ity in range(-delta, delta + 1):
                for itz in range(-delta, delta + 1):
                    # focal_channel[140-transpose[0], 140-transpose[1], 180-transpose[2]] = 1
                    if inR(itx, ity,itz):
                        focal_channel[x0+itx, y0+ity, z0+itz] = 1
        
        skull_ = torch.cat((skull_.unsqueeze(0), focal_channel.unsqueeze(0)))
        return skull_


# _________________________________________________________________________________________________________________
# DATASET: (x,y) = (256*256 data along rays, phases)
class SkullRayDS(SkullPhaseDS):
    ray_map = torch.tensor([])
    
    def __init__(
        self, 
        folder, 
        max_buf_len=100500, 
        valid=False, 
        skull_training=False,
        shuffle=False
    ):
        super().__init__(folder, max_buf_len, valid)
        self.skull_training = skull_training
        self.shuffle = shuffle
        
        # create ray map:
        SkullRayDS.__create_ray_map()
        
        # len == number of focuses * 256, if we training ray->ray to have not SO much points
        if not self.skull_training:
            self.len = self.len * 256

        # fire up to save all the skulls
        for ind in range(max_buf_len):
            try:
                self._read_file(30*ind)
            except:
                pass
        self.skull_keys = list(self.skulls.keys())
        
        # initialize some used further parameters
        self.initialize_buffers()
        self.debug_buf = []
        
    def __getitem__(self, ind):
    # Ordinary getter
        if self.skull_training:
            x, y = self.consecutive_data_getter(ind)
        else:
            x, y = self.random_data_getter(ind)
            x = x.unsqueeze(0)
        
        return x, y

    def consecutive_data_getter(self, ind):
    # Training on skulls requires obtaining all skull rays consecutively
        # load new focus:
        if self.shuffle:
            self.focus_ind = torch.randint(len(self.focuses), (1,)).int()
        else:
            self.focus_ind += 1
            self.focus_ind %= len(self.focuses)
        
        # load focal, phase and skull data:
        PHASE, focus_, skull_num, _ = self.focuses[self.focus_ind].get()
        skull_ = self.skulls[skull_num]
        # map shifting:
        self.ray_map_buf = SkullRayDS.__shift_ray_map(focus_.cuda()).int().cpu()
        # mapping skull on the tay map:
        X = skull_[self.ray_map_buf[:,:,:,0], self.ray_map_buf[:,:,:,1], self.ray_map_buf[:,:,:,2]] # .view(-1,256)

        return X, PHASE
    
    def random_data_getter(self, ind):
    # Training of AE requires random rays
        ind_ray = self.rgenerator(ind) #torch.randint(256*256, (1,)) % (256**2) #torch.tensor(ind % (256**2)).int()
        ind_ray_x = (ind_ray / 256).int()
        ind_ray_y = (ind_ray % 256).int()

        skull_keys = list(self.skulls.keys())
        skull_ind = torch.randint(len(self.skull_keys), (1,)).int()
        skull_ = self.skulls[skull_keys[skull_ind]]

        if self.ray_map_age % 256 == 0:
            focus_ = (torch.tensor((0,0,4/3*(ind % 30))) + 
                      torch.tensor( ((5* torch.randn(1)) % 30, (5* torch.randn(1)) % 30, (1 * torch.randn(1)).abs() % 5) )).ceil().int()

            ray_map_shifted = SkullRayDS.__shift_ray_map(focus_.cuda()).int() #focus_['transpose']
            self.ray_map_buf = ray_map_shifted
            self.shift_buf = focus_
        self.ray_map_age += 1
        
        ind_skull_x = self.ray_map_buf[ind_ray_x, ind_ray_y, :, 0].cpu()
        ind_skull_y = self.ray_map_buf[ind_ray_x, ind_ray_y, :, 1].cpu()
        ind_skull_z = self.ray_map_buf[ind_ray_x, ind_ray_y, :, 2].cpu()
        RAY = skull_[ind_skull_x, ind_skull_y, ind_skull_z]
        
        return RAY, -torch.ones(256)

    def initialize_buffers(self):
        self.ray_map_age = torch.tensor(0)
        self.ray_map_buf = None
        self.ray_skull_mapped = None
        self.focus_ind   = torch.tensor(0)
        self.shift_buf   = None
        self.random_dict = torch.tensor([])
        self.phase_buf   = torch.tensor([])
        
    def rgenerator(ind):
    # determined pseudo random - we want always to be able to get the ray number from ind:
        if self.random_dict.shape[0] < 200**2:
            self.random_dict = torch.randint(256**2, (241*227,)).int()
            while self.random_dict.unique().shape[0] < 200**2:
                self.random_dict[torch.randint(256**2, (1,)).int()] = torch.randint(256**2, (1,)).int()
        
        return self.random_dict[ind % self.random_dict.shape[0]]
    
    @staticmethod
    def __shift_ray_map(shift):
    # shifter for the ray map to match it with focal point:
        ray_map_shifted = SkullRayDS.ray_map + shift * torch.tensor((1, 1, -1)).to(shift.device)
        bool_cond = torch.zeros_like(ray_map_shifted).bool()
        
        bool_cond[:,:,:, 2] += (ray_map_shifted[:, :, :,  2]       > 235).bool()
        bool_cond[:,:,:,:2] += (ray_map_shifted[:, :, :, :2]       > 279).bool()
        bool_cond[:,:,:,: ] += (ray_map_shifted[:, :, :, : ]       <   0).bool()
        bool_cond = bool_cond.bool()
        
        ray_map_shifted[:, :, :,  2][(bool_cond[:,:,:,0].bool() + bool_cond[:,:,:,1].bool() + bool_cond[:,:,:,2].bool()).bool()] = torch.tensor((0)).to(torch.int16)
        ray_map_shifted[:, :, :, :2] = ray_map_shifted[:, :, :, :2] % 280
        return ray_map_shifted

    @staticmethod
    def __create_ray_map():
    # method to create ray map once and for all:
        if SkullRayDS.ray_map.shape != torch.Size([256, 256, 256, 3]):
            SkullRayDS.ray_map = torch.zeros(256,256,256,3)# fill constant map => ray_map += shift => ray_map inside [0,0,0; 280,280,236];
            
            tic = time.time()
            for indx in range(256):
                for indy in range(256):
                    SkullRayDS.ray_map[indx, indy, :] = SkullRayDS.__create_ray(indx-128, indy-128)
            toc = time.time()
            print('Map creation requires ', toc-tic, ' sec')

            SkullRayDS.ray_map = torch.tensor((140, 140, 0)) + SkullRayDS.ray_map
            SkullRayDS.ray_map[:, :, :, 2] = 180 - SkullRayDS.ray_map[:, :, :, 2]
            SkullRayDS.ray_map[:, :, :, 2][SkullRayDS.ray_map[:, :, :, 2] < 0] = 0

            SkullRayDS.ray_map[:, :, :, 2] = torch.clamp(SkullRayDS.ray_map[:, :, :, 2], min=0, max=235)
            
            SkullRayDS.ray_map = SkullRayDS.ray_map.cuda()
    
    @staticmethod
    def __create_ray(indx, indy):
    # method to create ray for ray map:
        if torch.tensor(indx).abs() > 128 or torch.tensor(indy).abs() > 128:
            raise RuntimeError('Wrong ray indexes: out of horizontal boundaries')
    
        alpha0 = torch.tensor(35 * torch.pi / 180)
        alpha_x = indx / 128 * alpha0
        alpha_y = indy / 128 * alpha0
    
        common_factor = (1 / torch.tensor(1 + alpha_x.tan().clone().detach() ** 2 + alpha_y.tan().clone().detach() ** 2).sqrt())
        dx = common_factor * alpha_x.tan()
        dy = common_factor * alpha_y.tan()
        dz = common_factor
    
        dr = torch.tensor((dx, dy, dz))
        ray = (torch.tensor((range(256), range(256), range(256))).t() * dr).floor().to(torch.int16)
    
        return ray