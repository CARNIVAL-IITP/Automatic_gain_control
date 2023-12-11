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
        self.pos_weight = args.pos_weight

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
            tr_avg_loss, tr_avg_acc = self._run_one_epoch(epoch, writer)
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
                torch.save(self.model.serialize(self.model,
                                                        self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss,
                                                       ),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)
            
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss, val_acc = self._run_one_epoch(epoch, writer, cross_valid=True)
            writer.add_scalar("train_acc", tr_avg_acc, epoch)
            writer.add_scalar("val_acc", val_acc, epoch)
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
                    torch.save(self.model.serialize(self.model,
                                                        self.optimizer, epoch + 1,
                                                        tr_loss=self.tr_loss,
                                                        cv_loss=self.cv_loss,
                                                        ),
                            self.model_path)
                
            

            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
        writer.flush()


    def _run_one_epoch(self, epoch, writer, cross_valid=False):
        start = time.time()
        total_loss = 0
        total_acc = 0
        total_true_pos = 0
        total_true_neg = 0
        total_false_pos = 0
        total_false_neg = 0
        summary = {"scalars" : {}}
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            wav, gt = data['wav'], data['label']
            gt = gt.float()
            if self.use_cuda:
                wav = wav.cuda()
                gt = gt.cuda()
            
            est_label = self.model(wav)
            pos_weight = torch.ones([est_label.shape[1]]) * self.pos_weight
            pos_weight.to('cuda')
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to('cuda')
            loss = criterion(est_label, gt)

            with torch.no_grad():
                acc = torch.sum(torch.where(torch.abs(est_label - gt) < 0.5, 1, 0)) / (gt.shape[0] * gt.shape[1])
                est_pos = torch.where(est_label > 0.5, 1, 0)
                gt_pos = torch.where(gt > 0.5, 1, 0)
                true_pos = torch.sum(est_pos * gt_pos)
                true_neg = torch.sum((1 - est_pos) * (1 - gt_pos))
                false_pos = torch.sum(est_pos * (1 - gt_pos))
                false_neg = torch.sum((1 - est_pos) * gt_pos)
            
            total_true_pos += true_pos
            total_true_neg += true_neg
            total_false_pos += false_pos
            total_false_neg += false_neg


            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.4f} | '
                      'Current Loss {3:.4f} | Average Accuracy {4:.4f} | TPR {5:.4f} | TNR {6:.4f} |{7:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), total_acc / (i + 1), total_true_pos / (total_true_pos + total_false_neg), total_true_neg / (total_true_neg + total_false_pos), 1000 * (time.time() - start) / (i + 1)), flush=True)
        if not cross_valid:
            summary['scalars'] = {
                'train_loss': total_loss / (i + 1)
            }
        else:
            summary['scalars'] = {
                'valid_loss': total_loss / (i + 1)
            }
        self.summarize(writer, epoch, scalars=summary['scalars'])

        return total_loss / (i + 1), total_acc / (i + 1)
    

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
    

