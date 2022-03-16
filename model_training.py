import torch
from torch import nn
import torch.nn.functional as F
# from modules import Res2d
import numpy as np
import random
import glob
random.seed(21)
import datetime
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vae import VAE
from data import EmovDB



class Trainer():
    def __init__(self, num_epochs, lr=0.001, loss_fn=None, reconstruction_term_weight=0.5, models_path='./models/'):
        self.epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.loss_fn = loss_fn
        self.models_path = models_path
        self.reconstruction_term_weight = reconstruction_term_weight
        print('running on ' + str(self.device))

    def train(self, model, trainLoader):
        if self.device.type == "cuda:0":
            torch.cuda.empty_cache()

        if self.loss_fn is None:
          loss_fn = nn.MSELoss(reduction='none')
        # criterion = nn.CrossEntropyLoss()
        # progress = []
        # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
        best_epoch, best_valAcc = 0, 0.6
        global_step, epoch = 0, 0
        start_time = time.time()

        if not isinstance(model, nn.DataParallel) and torch.cuda.device_count() > 3:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)#, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(model.parameters())
        
        print("start training...")
        history = []
        train_losses = []
        mini_batch_size=30
        print(datetime.datetime.now())
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        for epoch in range(self.epochs[0], self.epochs[1]):
            print('---------------')
            trainLoss = 0
            trainAcc = 0
            samples = 0
            model.train()
            running_loss = 0.0
            for i, (specs, _) in enumerate(trainLoader):
                specs = specs.to(self.device)
                encoded, z_mean, z_log_var, decoded, z = model(specs)
                # total loss = reconstruction loss + KL divergence
                # kl_divergence = (0.5 * z_mean^2 + torch.exp(z_log_var) - z_log_var - 1)).sum()

                kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1) # sum over latent dimension
                batch_size = kl_div.size(0)
                kl_div = kl_div.mean() # average over batch dimension
                
                pixelwise = loss_fn(decoded, specs)
                pixelwise = pixelwise.view(batch_size, -1).sum(axis=1) # sum over pixels
                pixelwise = pixelwise.mean() # average over batch dimension
               
                loss = self.reconstruction_term_weight * pixelwise + kl_div
                
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                trainLoss += loss.item() * batch_size
                samples += batch_size
              
                if i % mini_batch_size == mini_batch_size-1:    # print every 100 mini-batches
                    # self.reconstruction_term_weight += 0.05
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / mini_batch_size:.3f}')
                    running_loss = 0.0
                    print('mse = ', pixelwise)
                    print('kl div = ', kl_div)
                    f, axarr = plt.subplots(1,2)
                    orig_im = specs[0].detach().cpu().numpy().squeeze()
                    decoded_im = decoded[0].detach().cpu().numpy().squeeze()
                    axarr[0].imshow(orig_im)
                    axarr[1].imshow(decoded_im)
                    plt.show()
            model_name = self.models_path + f'model_{epoch+1}_{trainLoss / samples}.pth'
            # best_valAcc = (valAcc / samples)
            torch.save(model.state_dict(), model_name)


##########################
### Dataset and Dataloader
##########################
# hyper-params
BATCH_SIZE = 16
NUM_EPOCH = 100
LR = 0.001
RANDOM_SEED = 123
LOSS_FN = None
#################
### model loading ###
model = VAE(256)
load_model = False
current_epoch = 0
models_path = f'./saved_models_256/'
if load_model:
  model_names = glob.glob(models_path + '*.pth')
  model_indices = [int(num.split('_')[2]) for num in model_names]
  current_epoch = max(model_indices)
  latest_model = model_names[np.argmax(np.asarray(model_indices))]
  # Load state_dict
  model.load_state_dict(torch.load(latest_model))

train_dataset = EmovDB(mode='all')

dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

trainer = Trainer(num_epochs=[current_epoch, current_epoch+NUM_EPOCH], lr=LR, loss_fn=LOSS_FN, reconstruction_term_weight=15, models_path=models_path)
trainer.train(model, dataloader)
