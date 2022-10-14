from .weight_standardization import weight_standardization, remove_weight_standardization
from .vector_quantize import VectorQuantize, ResidualVQ
from .viterbi_vector_quantize import ViterbiVQ, ViterbiVQLegacy
from .scaled_activation import ScaledActivation


class EmptyScheduler:
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def step(self):
        pass

    def load_state_dict(self, state_dict):
        #if state_dict is not None:
        #    raise KeyError("Tried to load a non-empty scheduler to an empty scheduler")
        pass

    def state_dict(self):
        return None
