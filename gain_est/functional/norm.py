def normalize(x, normalize, mean, std):
    if normalize:
        x = (x - mean) / std
    return x

def denormalize(x, normalize, mean, std):
    if normalize:
        x = x * std + mean
    return x