import argparse
from dataset import AGCDataset#, collate
from solver import Solver
from model import AGC_STFT_GRU
import torch
from torch.utils.data import DataLoader
import config
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

parser = config.parser

def main(args):
    tr_dataset = AGCDataset(json_path = '/home/yhjeon/AGC/data/agc_1/train.json', sample_rate=args.sample_rate, wav_len=args.wav_len, win_len=args.win_len, hop_len=args.hop_len)
    tr_dataloader = DataLoader(tr_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    cv_dataset = AGCDataset(json_path = '/home/yhjeon/AGC/data/agc_1/valid.json', sample_rate=args.sample_rate, wav_len=args.wav_len, win_len=args.win_len, hop_len=args.hop_len)
    cv_dataloader = DataLoader(cv_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)

    data = {'tr_loader': tr_dataloader, 'cv_loader': cv_dataloader}

    model = AGC_STFT_GRU(480, 40, 480, 160)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1.0e-7)
    solver = Solver(data, model, optimizer, scheduler, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)