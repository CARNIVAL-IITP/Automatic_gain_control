from typing import Union, Dict
import torch
import numpy as np

MATPLOTLIB_FLAG = False


def join(name1: str, name2: str):
    if name1 != "":
        return f"{name1}.{name2}"
    else:
        return name2


def plot_param_and_grad(
    dict_to_plot: dict,
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    model_name: str = ""
) -> None:
    if isinstance(model, dict):
        for key, value in model.items():
            plot_param_and_grad(dict_to_plot, key, join(model_name, key))
        return
     
    if hasattr(model, "module"):
        model = model.module

    for param_name, param in model.named_parameters():
        dict_to_plot[f"param/{join(model_name, param_name)}"] = param.data.detach().cpu().numpy()
        if param.grad is None or torch.any(torch.isnan(param.grad.data)) or torch.any(torch.isinf(param.grad.data)):
            continue
        dict_to_plot[f"grad/{join(model_name, param_name)}"] = param.grad.data.detach().cpu().numpy()
    for buffer_name, buffer in model.named_buffers():
        if buffer.numel() == 0:
            continue
        if torch.any(torch.isnan(buffer.data)) or torch.any(torch.isinf(buffer.data)):
            continue
        dict_to_plot[f"buffer/{join(model_name, buffer_name)}"] = buffer.data.detach().cpu().numpy()


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt
    
    fig, ax = plt.subplots(figsize=(10,2))
    v = 0.0 if spectrogram.shape[0] == 80 else 2.0
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                    interpolation='none', vmin=-11.5+v, vmax=2.0+v)
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_numpy(nparray, scale="linear"):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(nparray)
    plt.yscale(scale)
    plt.xlabel("Index")
    plt.ylabel("Std")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def summarize(writer, epoch,  scalars={}, specs={}, images={}, audios={}, hists={}, sampling_rate=None, end='\n', print_=True):
    for key, value in scalars.items():
        writer.add_scalar(key, value, epoch)
        if print_:
            if type(value)==float:
                print(f"   {key}: {value:.4f}", end="")
            else:
                print(f"   {key}: {value}", end="")
    if scalars and print_:
        print("", end=end)
    
    for key, value in specs.items():
        img = plot_spectrogram_to_numpy(value)
        writer.add_image(key, img, epoch, dataformats='HWC')
    for k, v in hists.items():
        writer.add_histogram(k, v, epoch)
    for key, value in images.items():
        writer.add_image(key, value, epoch, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, epoch, sampling_rate)


if __name__=="__main__":
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="logs")    # ./logs 디렉토리에 로그 생성

    for epoch in range(100):
        # 매 epoch마다 loss, audio, mel 얻는다고 가정
        audio = torch.rand(480000)      # 48000 길이의 audio
        mel = torch.randn(15, 20)     # 15 x 20 크기의 mel spectrogram

        scalars = {"loss_mel": 0.01, "loss_G": 1.2, "loss_D": 0.3}
        spec = {"mel": mel.squeeze().cpu().numpy()}
        audio = {"audio": audio.squeeze().cpu().numpy()}
        summarize(writer, epoch, scalars=scalars, specs=spec, audios=audio)
