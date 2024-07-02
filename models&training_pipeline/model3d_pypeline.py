import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class LightningModel(pl.LightningModule):
    def __init__(self, model, lr=1e-6, loss_combination=[1,0], loss_foo=None, scheduler=None, optimizer=None, folder="validation/"):
        ## scheduler must be either *LAMBDA optimizer: scheduler(optimizer, PARAMETERS)* or *None*
        ## optimizer must be either *LAMBDA model_parameters: optimizer(model_parameters, PARAMETERS)* or *None*
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.loss_combination = torch.tensor(loss_combination, dtype=torch.float) / torch.norm(torch.tensor(loss_combination, dtype=torch.float))
        if loss_foo != None:
            self.loss_foo = lambda y_hat, y: loss_foo(y_hat, y)
        else:
            self.loss_foo = lambda y_hat, y: self.phase_l1(y_hat, y)*self.loss_combination[0] + self.phase_l2(y_hat, y)*self.loss_combination[1]

        self.scheduler = scheduler
        self.loss_out_dict = {}
        self.folder = folder
        if len(self.folder) > 0:
            if self.folder[-1] != '/':
                self.folder = self.folder + '/'
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.optimizer != None:
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # x, y = x_test, y_test
        y_hat = self.model(x)

        # loss = f(l1, l2):
        l1_loss = self.phase_l1(y_hat, torch.cos(y / 2))
        l2_loss = self.phase_l2(y_hat, torch.cos(y / 2))
        loss    = self.loss_foo(y_hat, torch.cos(y / 2))
        # log losses:
        self.log('train_loss',       loss, prog_bar=True,  on_epoch=True, on_step=True)
        self.log('train_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=True)
        self.log('train_l2_loss', l2_loss, prog_bar=False, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        y_hat = self.model(x)

        # loss = f(l1, l2):
        l1_loss = self.phase_l1(y_hat, torch.cos(y))
        l2_loss = self.phase_l2(y_hat, torch.cos(y))
        loss    = self.loss_foo(y_hat, torch.cos(y))
        # log losses:
        self.log('valid_loss',       loss, prog_bar=True,  on_epoch=True, on_step=True)
        self.log('valid_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=True)
        self.log('valid_l2_loss', l2_loss, prog_bar=False, on_epoch=True, on_step=True)
        self.loss_out_dict[loss] = {'x': x.detach().cpu().to(torch.float32).numpy(), 'pred': y_hat.detach().cpu().to(torch.float32).numpy()}
        
        return [loss, x, y_hat]

    # def validation_step_end(self, validation_step_outputs):
    #     loss, x, y_hat = validation_step_outputs
    #     self.loss_out_dict[loss] = {'x': x.detach().cpu().numpy(), 'pred': y_hat.detach().cpu().numpy()}

    def on_validation_epoch_end(self):
        try:
            max_loss = max(self.loss_out_dict.keys())
            min_loss = min(self.loss_out_dict.keys())
    
            out = {}
            out['max_loss'] = self.loss_out_dict[max_loss]
            out['min_loss'] = self.loss_out_dict[min_loss]
            hdf5storage.savemat(file_name=self.folder + f"validation_outputs_epoch_{self.current_epoch:02d}.mat", 
                                mdict=out, 
                                format='7.3')
        except Exception as e:
            print(e)
            # print('something wrong, I can feel it')
        self.loss_out_dict = {}
        
    @staticmethod
    def phase_l2(y, y_hat):
        return F.mse_loss(y, y_hat)
    @staticmethod
    def phase_l1(y, y_hat):
        return F.l1_loss(y, y_hat)


def create_checkpoint_callbacks(project, run_name, save_top_k=3):
    # saves top-K checkpoints based on "valid_l1_loss" metric
    checkpoint_callback_l1 = ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="valid_l1_loss",
        mode="min",
        dirpath= project + "/" + run_name + "/",
        filename=project + "-{epoch:02d}-{valid_l1_loss:.2f}",
    )
    # saves top-K checkpoints based on "valid_l2_loss" metric
    checkpoint_callback_l2 = ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="valid_l2_loss",
        mode="min",
        dirpath= project + "/",
        filename=project + "-{epoch:02d}-{valid_l2_loss:.2f}",
    )
    return [checkpoint_callback_l1, checkpoint_callback_l2]


def train_3d_model(project, model,
                   train_loader, valid_loader,
                   LR, LOSS_FUNCTIONS, OPTIMIZERS, optim_names,
                   max_epochs=15, precision="16-mixed"):
## project - project name; model - nn.Model class; loaders; 
## LR - [number of learning rates to check]; LOSS_FUNCTIONS - [number of loss functions to check]; OPTIMIZERS - [number of optimizers to check]; optim_names - ["opimizers", "names"];
## max_epochs - literally; precision - "16-mixed" for tensor cores / 32.
    ind = 0
    ind_optim = 0
    
    for lr in LR:
        for optim in OPTIMIZERS:
            ind_optim += 1
            for loss_foo in LOSS_FUNCTIONS:
                ind += 1
                model = None
                lightning_model = None
            
                # logger setup
                if 'wandb_logger' in locals():
                    del wandb_logger
                    wandb.finish()
        
                run_name=f'run#{ind:d}, {lr}, no scheduler, , {loss_foo.__name__}, {optim_names[ind_optim-1]}'
                wandb_logger = WandbLogger(project=project, name=run_name)
        
                # Lightning model&trainer
                lightning_model = LightningModel(model, lr=lr, loss_foo=loss_foo, folder=project + '/' + run_name + '/')
                trainer = pl.Trainer(max_epochs=15, logger=wandb_logger, precision="16-mixed",
                                     check_val_every_n_epoch=1, callbacks=create_checkpoint_callbacks(project, run_name))
            
                trainer.fit(lightning_model, train_loader, valid_loader)