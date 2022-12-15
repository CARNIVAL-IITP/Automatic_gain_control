import argparse
from dataset import VADDataset#, collate
from solver import Solver
from model import VAD
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser("Neural voice activity detection for automatic gain control")

# General settings
parser.add_argument('--json_path', type=str, default=None,
                    help='directory including train.json, test.json')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='sampling rate of waveform datasets')
parser.add_argument('--segment', default=4., type=float,
                    help='train segment length (in seconds)')
parser.add_argument('--name', type=str,
                    help='Name of the experiment')


# Training configuration
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='Use GPU for training')
parser.add_argument('--epochs', default=20, type=int,
                    help='Number of training epochs')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='# of batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Initial learning rate')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
parser.add_argument('--save_folder', default='./logs/ckpts',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')

def main(args):
    tr_dataset = VADDataset(json_path = args.json_path, mode='train', 
                            sample_rate = args.sample_rate, frame_size=25)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cv_dataset = VADDataset(json_path = args.json_path, mode='test',  
                            sample_rate = args.sample_rate, frame_size=25)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    data = {'tr_loader': tr_dataloader, 'cv_loader': cv_dataloader}

    model = VAD(512, 80, 400, 400)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800, eta_min=1.0e-7)
    solver = Solver(data, model, optimizer, scheduler, args)
    solver.train()

if __name__=='__main__':
    args = parser.parse_args()
    print(args)
    main(args)
