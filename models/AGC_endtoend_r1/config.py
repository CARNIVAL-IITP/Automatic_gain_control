import argparse

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

parser.add_argument('--wav_len', default=160000, type=int, help='Wave segment length (in samples)')
parser.add_argument('--win_len', default=160, type=int, help='Conv1d kernel length (in samples)')
parser.add_argument('--hop_len', default=80, type=int, help='Conv1d stride length (in samples)')


# Training configuration
parser.add_argument('--print_freq', default=100, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='Use GPU for training')
parser.add_argument('--epochs', default=200, type=int,
                    help='Number of training epochs')
parser.add_argument('--batch_size', default=128, type=int,
                    help='# of batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
parser.add_argument('--lr', default=1e-2, type=float,
                    help='Initial learning rate')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
parser.add_argument('--save_folder', default='/home/yhjeon/projects/IITP_SE/NS_AGC/models/AGC_endtoend_r1/ckpts/h40',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_model', default='/home/yhjeon/projects/IITP_SE/NS_AGC/models/AGC_endtoend_r1/ckpts/h40/best.pth.tar',
                    help='Continue from checkpoint model')
parser.add_argument('--continue_from', default=False,
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='/home/yhjeon/projects/IITP_SE/NS_AGC/models/AGC_endtoend_r1/ckpts/h40/best.pth.tar',
                    help='Location to save best validation model')