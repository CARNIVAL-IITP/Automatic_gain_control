import argparse
from dataset import VADDataset#, collate
from solver import Solver
from self_attentive_vad import SelfAttentiveVAD
import torch
from torch.utils.data import DataLoader
import config
import matplotlib.pyplot as plt

parser = config.parser

def main(args):
    tt_dataset = VADDataset(json_path = '/home/yhjeon/AGC/src/vad/test_clean.json', sample_rate=args.sample_rate, wav_len=args.wav_len, win_len=args.win_len, hop_len=args.hop_len)
    tt_dataloader = DataLoader(tt_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)

    data = {'tt_loader': tt_dataloader}

    model = SelfAttentiveVAD(args.win_len, args.hop_len, 32, 32, 1, 0)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    package = torch.load('/home/yhjeon/AGC/final.pth.tar')
    model.module.load_state_dict(package['state_dict'])
    model.eval()

    threshold = 10/11

    for i, data in enumerate(tt_dataloader):

        wav, gt = data['wav'], data['label']
        gt = gt.float()
        wav = wav.cuda()
        gt = gt.cuda()
        est_label = model(wav)

        est_label = torch.where(est_label > threshold, 1, 0)
        plt.plot(est_label)
        plt.plot(gt)
        plt.show()



   
    

if __name__ == '__main__':
    config = parser.parse_args()
    print(config)
    main(config)