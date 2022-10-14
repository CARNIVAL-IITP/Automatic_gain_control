from .hparams import get_hparams, HParams
from .summarize import summarize, plot_param_and_grad, plot_numpy
from .grad_clip import clip_grad_norm_local

# for debugging
from .debug import Printer, check_grad