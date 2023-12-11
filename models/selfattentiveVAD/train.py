import argparse
from dataset import VADDataset#, collate
from solver import Solver
from self_attentive_vad import SelfAttentiveVAD, SelfAttentiveVAD_Normalize, SelfAttentiveVAD_Normalize_Convoutput
import torch
from torch.utils.data import DataLoader
import config

parser = config.parser

def main(args):
    tr_dataset = VADDataset(json_path = '/home/yhjeon/AGC/src/vad/train.json', sample_rate=args.sample_rate, wav_len=args.wav_len, win_len=args.win_len, hop_len=args.hop_len)
    tr_dataloader = DataLoader(tr_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    cv_dataset = VADDataset(json_path = '/home/yhjeon/AGC/src/vad/valid.json', sample_rate=args.sample_rate, wav_len=args.wav_len, win_len=args.win_len, hop_len=args.hop_len)
    cv_dataloader = DataLoader(cv_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)

    data = {'tr_loader': tr_dataloader, 'cv_loader': cv_dataloader}

    # model = SelfAttentiveVAD(args.win_len, args.hop_len, 32, 32, 8, 0)
    # model = SelfAttentiveVAD_Normalize(args.win_len, args.hop_len, 32, 32, 2, 0)
    model = SelfAttentiveVAD_Normalize_Convoutput(args.win_len, args.hop_len, 16, 16, 1, 0)
    if args.use_cuda:
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800, eta_min=1.0e-7)
    solver = Solver(data, model, optimizer, scheduler, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)