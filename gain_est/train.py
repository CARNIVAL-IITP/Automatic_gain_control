import os, time, argparse, random, socket, atexit, warnings

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.data_audio import Dataset, DatasetPreprocessed, DNSDataset, AECDataset, collate
from utils import get_hparams, summarize
from models import get_wrapper
import random


def close_writer(writer : SummaryWriter):
    writer.close()


def main():
    
    assert torch.cuda.is_available(), "CPU training is not allowed."

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, type=str, help="checkpoints and logs will be saved at logs/{name}")
    parser.add_argument('-c', '--config', required=True, type=str, help="path to config json file")

    a = parser.parse_args()

    base_dir = os.path.join('logs', a.name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    hps = get_hparams(a.config, base_dir, save=True)
    
    ### seeding ###
    print(f'seed : {hps.train.seed}')
    torch.manual_seed(hps.train.seed)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(hps.train.seed)
    random.seed(hps.train.seed)
    ###############
    
    
    # assign available port
    os.environ['MASTER_ADDR'] = 'localhost'
    sock = socket.socket()
    sock.bind(('localhost', 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    os.environ['MASTER_PORT'] = str(sock.getsockname()[1])
    sock.close()

    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs: {n_gpus}\n')
    print(hps.model)
    
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    else:
        run(0, n_gpus, hps)


def run(rank, n_gpus, hps):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    hp = hps.train

    seed = hp.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(rank)
    wrapper = get_wrapper(hps.model)(hps, train=True, rank=rank)
    wrapper.load()
    
    # if set persistent_workers to True, train_loader & valid_loader will be persistent.
    # It may speed up training (save time for spawn), but you'd better set num_workers small enough.
    persistent_workers = getattr(hp, "persistent_workers", False)
    pin_memory = getattr(hp, "pin_memory", False)
    if pin_memory and persistent_workers:
        warnings.warn(
            f"In PyTorch<=1.7.1, setting persistent_workers=True & pin_memory=True will cause RuntimeError. Your PyTorch version: {torch.__version__}",
            RuntimeWarning
        )
    
    batch_size = hp.batch_size * n_gpus
    batch_size_valid = getattr(hp, "batch_size_valid", hp.batch_size) * n_gpus
    batch_size_infer = getattr(hp, "batch_size_infer", 1)
    if hasattr(hps.data, "clean_dir"):
        _Dataset, _collate = DNSDataset, None
    elif hasattr(hps.data, "near_dir"):
        _Dataset, _collate = AECDataset, None
    elif hps.data.data_dir == "":
        _Dataset, _collate = Dataset, collate
    else:
        _Dataset, _collate = DatasetPreprocessed, collate
    train_dataset = _Dataset(hps.data, wrapper.keys, wrapper.textprocessor, mode="train",
        batch_size=batch_size, verbose=(rank==0))
    train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=hp.batch_size,
        shuffle=False, num_workers=hp.num_workers, collate_fn=_collate,
        persistent_workers=persistent_workers, pin_memory=pin_memory)
    val_dataset = _Dataset(hps.data, wrapper.keys, wrapper.textprocessor, mode="valid",
        batch_size=batch_size_valid, verbose=(rank==0))
    val_sampler = DistributedSampler(val_dataset, num_replicas=n_gpus, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size_valid,
        shuffle=False, num_workers=hp.num_workers, collate_fn=_collate,
        persistent_workers=persistent_workers, pin_memory=pin_memory)
    
    if rank == 0:
        infer_dataset = _Dataset(hps.data, wrapper.infer_keys, wrapper.textprocessor, mode="infer")
        # spawn may takes longer. Therefore, we set num_workers=0
        infer_loader = DataLoader(infer_dataset, batch_size=batch_size_infer, shuffle=False, num_workers=0,
            collate_fn=_collate)

        writer_train = SummaryWriter(log_dir=os.path.join(hps.base_dir, "train"))
        writer_valid = SummaryWriter(log_dir=os.path.join(hps.base_dir, "valid"))
        atexit.register(close_writer, writer_train)
        atexit.register(close_writer, writer_valid)
    
        if wrapper.epoch == 0:
            if wrapper.plot_param_and_grad:
                hists = wrapper.plot_initial_param(train_loader)
                summarize(writer_train, epoch=0, hists = hists)
            #wrapper.save()
        
        start_time = time.time()

    
    for epoch in range(wrapper.epoch + 1, hps.train.max_epochs + 1):
        wrapper.epoch = epoch
        lr = wrapper.get_lr()
        
        # train
        train_dataset.shuffle(hp.seed + epoch)
        summary_train = wrapper.train_epoch(train_loader)
        
        # valid
        summary_valid = wrapper.valid_epoch(val_loader)
        
        # summarize & infer
        if rank == 0:
            '''
            if epoch == 1 or epoch % hp.infer_interval == 0:
                
                summary_infer = wrapper.infer_epoch(infer_loader)
                summarize(writer_valid, epoch, sampling_rate = hps.data.sampling_rate, **summary_infer)
            
            '''
            if epoch % hp.save_interval == 0:
                wrapper.save()

            end_time = time.time()
            scale = wrapper.scaler.get_scale()
            if "scalars" not in summary_train:
                summary_train["scalars"] = {}
            summary_train["scalars"]["lr"] = lr
            summary_train["scalars"]["scale"] = scale
            print(f"Epoch {epoch} - Time: {end_time - start_time:.1f} sec\tName: {hps.base_dir}")
            print("\tTrain", end="")
            summarize(writer_train, epoch, **summary_train)
            print("\tValid", end="")
            summarize(writer_valid, epoch, **summary_valid)
            start_time = end_time
    
    if rank == 0:
        writer_train.close()
        writer_valid.close()
    #dist.destroy_process_group()


if __name__ == '__main__':
    main()
