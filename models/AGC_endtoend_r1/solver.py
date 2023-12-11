import os
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

writer = SummaryWriter()

class Solver(object):
    def __init__(self, data, model, optimizer, scheduler, args):
        self.data = data
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.max_norm = args.max_norm
        self.print_freq = args.print_freq

        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.continue_model = args.continue_model
        self.model_path = args.model_path

        self._reset()
    
    def _reset(self):
        if self.continue_from:
            print('Loading checkpoint model {}'. format(self.continue_model))
            package = torch.load(self.continue_model)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float('inf')
        self.best_val_loss = float('inf')

    # todo: train 차분히 읽어보기
    def train(self):
        os.makedirs('./logs/' + str(self.args.name), exist_ok=True)
        writer = SummaryWriter(log_dir="logs/{}".format(self.args.name))
        for epoch in range(self.start_epoch, self.epochs):
            print('Training...')
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch, writer)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.6f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85) 
            self.scheduler.step()
            optim_state = self.optimizer.state_dict()
            self.optimizer.load_state_dict(optim_state)

            print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))

            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(model=self.model.module, optimizer=self.optimizer, epoch=epoch + 1,tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print('Saving checkpoint model to %s' % file_path)
            
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, writer, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.6f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            self.prev_val_loss = val_loss

            # Save the best model

            if self.prev_val_loss < self.best_val_loss:
                self.best_val_loss = self.prev_val_loss
                if epoch > 0:
                    print('Saving best model to %s' % self.model_path)
                    torch.save(self.model.module.serialize(model=self.model.module, optimizer=self.optimizer, epoch=epoch + 1, tr_loss=self.tr_loss, cv_loss=self.cv_loss), self.model_path)
                
            

            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
        writer.flush()


    def _run_one_epoch(self, epoch, writer, cross_valid=False):
        start = time.time()
        total_loss = 0
        summary = {"scalars" : {}}
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            wav, gt = data['distorted'], data['original']
            gt = gt.float()
            if self.use_cuda:
                wav = wav.cuda()
                gt = gt.cuda()
            
            est, _ = self.model(wav)
            criterion = torch.nn.MSELoss()
            loss = criterion(torch.log(torch.abs(est) + 1e-5), torch.log(torch.abs(gt) + 1e-5))

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()
            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.4f} | '
                      'Current Loss {3:.4f} |{4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)), flush=True)
        if not cross_valid:
            summary['scalars'] = {
                'train_loss': total_loss / (i + 1)
            }
        else:
            summary['scalars'] = {
                'valid_loss': total_loss / (i + 1)
            }
        self.summarize(writer, epoch, scalars=summary['scalars'])

        return total_loss / (i + 1)
    

    def summarize(self, writer, epoch,  scalars={}, specs={}, images={}, audios={}, hists={}, sampling_rate=16000, end='\n'):
        for key, value in scalars.items():
            writer.add_scalar(key, value, epoch)
            if type(value)==float:
                print(f"   {key}: {value:.4f}", end="")
            else:
                print(f"   {key}: {value}", end="")
        if scalars:
            print("", end=end)
        '''
        for key, value in specs.items():
            img = plot_spectrogram_to_numpy(value)
            writer.add_image(key, img, epoch, dataformats='HWC')
        for k, v in hists.items():
            writer.add_histogram(k, v, epoch)
        '''
        for key, value in images.items():
            writer.add_image(key, value, epoch, dataformats='HWC')
        
        for k, v in audios.items():
            writer.add_audio(k, v, epoch, sampling_rate)