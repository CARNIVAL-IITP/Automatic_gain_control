import torch
import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import as_strided
from numba import jit


# dictionary of (n_fft, sampling frequency): (critical band, Tuple(k_start, k_end, j_start, j_end))
NFFT_FS_TO_CB_K_J = {
    # Layer I
    "512_16000": (21, ((3, 96, 2, 2), (96, 192, 2, 3), (192, 251, 2, 6))),
    "512_22050": (23, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 251, 2, 6))),
    "512_24000": (23, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 251, 2, 6))),
    "512_32000": (24, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 251, 2, 6))),
    "512_44100": (25, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 251, 2, 6))),
    "512_48000": (25, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 251, 2, 6))),
    # Layer II
    "1024_16000": (21, ((5, 192, 4, 4), (192, 384, 2, 6), (384, 501, 2, 12))),
    "1024_22050": (23, ((5, 128, 4, 4), (128, 256, 2, 6), (256, 501, 2, 12))),
    "1024_24000": (23, ((5, 128, 4, 4), (128, 256, 2, 6), (256, 501, 2, 12))),
    "1024_32000": (24, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 256, 2, 6), (256, 501, 2, 12))),
    "1024_44100": (25, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 256, 2, 6), (256, 501, 2, 12))),
    "1024_48000": (25, ((3, 64, 2, 2), (64, 128, 2, 3), (128, 256, 2, 6), (256, 501, 2, 12))),
}

# absolute threshold of hearing. Shape: [n_fft // 2 + 1]
Tq = {}

# Filter bank of critical band. shape: [n_fft // 2 + 1, critical_band]
CFB = {}

# mapping of frequency_bin to bark_scale(critical band). Shape: [critical_band]
Z = {}

# frequency_bin of non_tonal maskers
# <=> geometric mean of frequency_bins within a critical band. Shape: [critical_band]
NON_TONAL_IDX = {}


def local_maximum(X: ndarray, k_start: int, k_end: int):
    BT, C = X.shape[0], X.shape[1]
    C_ = k_end - k_start
    x_1 = as_strided(X.ravel()[k_start-1:], shape=(BT, C_, 2), strides=(4*C, 4, 8))
    is_local_maximum = np.zeros_like(X, dtype=bool)
    np.greater(np.expand_dims(X[:, k_start:k_end], 2), x_1).all(axis=2, out=is_local_maximum[:, k_start:k_end])
    return is_local_maximum


def tonal_mask_(X: ndarray, X_tm: ndarray, X_for_nm: ndarray, is_tonal: ndarray, k_start: int, k_end: int, j_start: int, j_end: int) -> None:
    """check X[k] - X[k+j] >= 7dB where j= -j_end + 1, ..., -j_start, j_start, ..., j_end - 1 for k_start <= k < k_end"""
    # check X[k] - X[k+j] >= 7 & update tonal_idx and X_for_nm
    BT, C = X.shape[0], X.shape[1]
    C_ = k_end - k_start
    x_orig = np.expand_dims(X, 2)[:, k_start:k_end, :]    # [BT, C_, 1]
    
    x_j = as_strided((X + 7.0).ravel()[k_start-j_end:], shape=(BT, C_, j_end-j_start+1, 2), strides=(C*4, 4, 4, (j_end+j_start)*4))
    is_tonal_ = np.empty((BT, C_, j_end-j_start+1, 2), dtype=bool)
    np.greater_equal(np.expand_dims(x_orig, 3), x_j, out=is_tonal_)
    is_tonal_ = is_tonal_.reshape(BT, C_, -1).all(axis=2)         # [BT, C_]
    np.logical_and(is_tonal[:, k_start:k_end], is_tonal_, out=is_tonal[:, k_start:k_end])   # [BT, C_]
    
    is_tonal_ = is_tonal[:, k_start:k_end]
    minus_inf = np.where(
        is_tonal_,
        -np.inf,
        np.zeros(1, dtype=np.float32)
    )
    minus_inf = np.expand_dims(minus_inf, 2)
    
    x_1 = as_strided(X_for_nm.ravel()[k_start-1:], shape=(BT, C_, 3), strides=(C*4, 4, 4))
    np.add(x_1, minus_inf, out=x_1)
    
    minus_inf = np.expand_dims(minus_inf, 3)
    x_j = as_strided(X_for_nm.ravel()[k_start-j_end:], shape=(BT, C_, j_end-j_start+1, 2), strides=(C*4, 4, 4, (j_end+j_start)*4))
    np.add(x_j, minus_inf, out=x_j)
    
    # calculate X_tm
    x_1 = as_strided(X.ravel()[k_start-1:], shape=(BT, C_, 3), strides=(C*4, 4, 4))
    x_tm = 10.0 * np.log10((10.0 ** (x_1 / 10.0)).sum(axis=2))
    
    X_tm[:, k_start:k_end] = np.where(
        is_tonal_,
        x_tm,
        -np.inf
    )


def tonal_mask(X: ndarray, X_tm: ndarray, X_for_nm: ndarray, is_tonal: ndarray, k_start: int, k_end: int, j_start: int, j_end: int) -> None:
    """check X[k] - X[k+j] >= 7dB where j= -j_end, ..., -j_start, j_start, ..., j_end for k_start <= k < k_end
    Use torch operation because it can use in-place operation"""
    X, X_tm, X_for_nm, is_tonal = torch.from_numpy(X), torch.from_numpy(X_tm), torch.from_numpy(X_for_nm), torch.from_numpy(is_tonal)
    BT, C = X.size(0), X.size(1)
    C_ = k_end - k_start
    x_orig = X.unsqueeze(2)[:, k_start:k_end, :]    # [BT, C_, 1]
    
    x_j = torch.as_strided(X + 7.0, size=(BT, C_, j_end-j_start+1, 2), stride=(C, 1, 1, j_end+j_start), storage_offset=k_start-j_end)
    is_tonal_ = torch.empty(BT, C_, j_end-j_start+1, 2, dtype=torch.bool)
    torch.ge(x_orig.unsqueeze(3), x_j, out=is_tonal_)
    is_tonal_ = is_tonal_.view(BT, C_, -1).all(dim=2)         # [BT, C_]
    torch.logical_and(is_tonal[:, k_start:k_end], is_tonal_, out=is_tonal[:, k_start:k_end])   # [BT, C_]
    
    is_tonal_ = is_tonal[:, k_start:k_end]
    zeros = torch.zeros(1, device=X.device).expand(BT, C_)
    minus_inf = torch.where(
        is_tonal_,
        torch.tensor(-float('inf'), device=X.device).expand(BT, C_),
        zeros
    )
    minus_inf = minus_inf.unsqueeze(2)
    
    x_1 = torch.as_strided(X_for_nm, size=(BT, C_, 3), stride=(C, 1, 1), storage_offset=k_start-1)
    x_1.add_(minus_inf)
    
    minus_inf = minus_inf.unsqueeze(3)
    x_j = torch.as_strided(X_for_nm, size=(BT, C_, j_end-j_start+1, 2), stride=(C, 1, 1, j_end+j_start), storage_offset=k_start-j_end)
    x_j.add_(minus_inf)
    
    # calculate X_tm
    x_1 = torch.as_strided(X, size=(BT, C_, 3), stride=(C, 1, 1), storage_offset=k_start-1)
    x_tm = 10.0 * torch.log10((10.0 ** (x_1 / 10)).sum(dim=2))
    X_tm[:, k_start:k_end] = torch.where(is_tonal_, x_tm, torch.tensor(-float('inf'), device=X.device).expand(BT, C_))


def non_tonal_mask(X_for_nm: ndarray, cfb: ndarray) -> None:
    # X_for_nm: [BT, n_fft//2+1]
    # CFB: [n_fft//2+1, critical_band]
    x = 10.0 ** (X_for_nm / 10.0)
    X_nm = np.matmul(x, cfb)
    return 10.0 * np.log10(np.clip(X_nm, a_min=1e-7, a_max=None))


@jit(nopython=True)
def deceimate_loop(X_tm: ndarray, is_tonal: int, z: ndarray):
    BT, C = X_tm.shape[0], X_tm.shape[1]  # C = n_fft//2+1
    
    # decimate tonal maskers which are within 0.5 bark scale
    for bt in range(BT):
        for idx in range(C):
            # Decimation ordering is the amplitude of the masker.
            # 1. Find the strongest masker and decimate maskers within 0.5 bark scale
            # 2. Find the second strongest masker among remaining maskers and decimate
            # 3. Find the third strongest masker among remaining maskers and decimate
            # 4. ... repeat until there's no remaining masker
            X_tm_ordered_idx = np.argsort(-X_tm[bt, :])
            k = X_tm_ordered_idx[idx]   # k is the frequency index of the idx-th strongest masker
            if is_tonal[bt, k] == 0:
                break
            
            # find tonal components within 0.5 bark scale and decimate.
            z1 = z[k]
            for j in range(k-1, -1, -1):
                if z1 - z[j] >= 0.5:
                    break
                if not is_tonal[bt, j]:
                    continue
                X_tm[bt, j] = -np.inf
                is_tonal[bt, j] = 0
            for j in range(k+1, C):
                if z[j] - z1 >= 0.5:
                    break
                if not is_tonal[bt, j]:
                    continue
                X_tm[bt, j] = -np.inf
                is_tonal[bt, j] = 0


def decimate(X_tm: ndarray, is_tonal: ndarray, X_nm: ndarray, non_tonal_idx: ndarray, Tq: ndarray, z: ndarray):
    # X_tm, is_tonal: [BT, n_fft//2+1]
    # X_nm, non_tonal_idx: [BT, critical_band]
    # Tq, z: [n_fft//2+1]
    
    # decimate tonal masker which is smaller than Tq
    is_tonal = np.where(
        X_tm >= np.expand_dims(Tq, axis=0),
        is_tonal.astype(np.float32),
        0
    )
    X_tm = np.where(
        X_tm >= np.expand_dims(Tq, axis=0),
        X_tm,
        -np.inf
    )
    
    # decimate non_tonal(noise) masker which is smaller than LTq
    Tq_ = np.expand_dims(Tq[non_tonal_idx[0, :]], 0)
    X_nm = np.where(
        X_nm >= Tq_,
        X_nm,
        -np.inf
    )
    non_tonal_idx = np.where(
        X_nm >= Tq_,
        non_tonal_idx,
        -1
    )
    deceimate_loop(X_tm, is_tonal, z)
    
    return X_tm, is_tonal, X_nm, non_tonal_idx, z


def spreading_function_numpy_op(k: int, X: ndarray, z: ndarray) -> ndarray:
    dz = z[k] - z
    X = np.where(dz >= -1, (0.4*X+6)*dz, 17.0*dz-0.4*X+11)
    X = np.where(dz >= 0, -17*dz, X)
    X = np.where(dz >= 1, (0.15*X-17)*dz-0.15*X, X)
    X = np.where(
        np.logical_or(dz < -3, dz >= 8),
        -np.inf,
        X
    )
    return X


def spreading_function(*args, **kwargs) -> ndarray:
    return spreading_function_numpy_op(*args, **kwargs)


@jit(nopython=True)
def global_mask(X_tm: ndarray, is_tonal: ndarray, X_nm: ndarray, non_tonal_idx: ndarray, Tq: ndarray, non_tonal_idx_global: ndarray, z: ndarray) -> ndarray:
    # X_tm, is_tonal: [BT, n_fft//2+1], cpu
    # X_nm, non_tonal_idx, non_tonal_idx_global: [BT, critical_band], cpu
    # Tq: [n_fft//2+1], cuda
    # z: [n_fft//2+1], cpu
    C = X_tm.shape[1]    # C = n_fft//2+1
    Tg = (10.0 ** (0.1 * Tq)).repeat(X_tm.shape[0]).reshape(-1, X_tm.shape[0]).transpose()
    Tg = np.ascontiguousarray(Tg)
    for bt in range(X_tm.shape[0]):
        nti = 0
        for k in range(C):
            if (is_tonal[bt, k] == 0) and (k != non_tonal_idx_global[nti]):
                continue
            
            if is_tonal[bt, k]:
                X = X_tm[bt, k]
                dz = z[k] - z
                X = np.where(dz >= -1, (0.4*X+6)*dz, 17.0*dz-0.4*X+11)
                X = np.where(dz >= 0, -17*dz, X)
                X = np.where(dz >= 1, (0.15*X-17)*dz-0.15*X, X)
                X = np.where(
                    np.logical_or(dz < -3, dz >= 8),
                    -np.inf,
                    X
                )
                
                Tg[bt, :] += 10.0 ** (0.1 * (X + (X_tm[bt, k] - 6.025 - 0.257 * z[k])))
            if k == non_tonal_idx_global[nti]:
                if non_tonal_idx[bt, nti] >= 0:
                    X = X_nm[bt, nti]
                    dz = z[k] - z
                    X = np.where(dz >= -1, (0.4*X+6)*dz, 17.0*dz-0.4*X+11)
                    X = np.where(dz >= 0, -17*dz, X)
                    X = np.where(dz >= 1, (0.15*X-17)*dz-0.15*X, X)
                    X = np.where(
                        np.logical_or(dz < -3, dz >= 8),
                        -np.inf,
                        X
                    )
                    Tg[bt, :] += 10.0 ** (0.1 * (X + (X_nm[bt, nti] - 2.025 - 0.175 * z[k])))
                if nti < non_tonal_idx_global.shape[0] - 1:
                    nti += 1
                    
    Tg = 10.0 * np.log10(Tg)
    return Tg


def absolute_threshold_of_hearing(n_fft: int, fs: float):
    delta = fs / n_fft
    f = np.arange(0, fs // 2 + delta, delta, dtype=np.float32)
    f[0] = 1e-7
    tq = np.minimum(
        3.64 * (f / 1000)**(-0.8) - 6.5 * np.e ** (-0.6 * (f / 1000 - 3.3)**2) + 10**(-3) * (f / 1000)**4,
        68.0
    )
    tq[0] = 68.0
    return tq   # spl(dB) scale


def global_masking_threshold(mag_spec, n_fft, fs):
    # mag_spec: [Batch, n_fft//2+1, Time]
    global Tq, CFB, Z, NON_TONAL_IDX
    key = f"{n_fft}_{fs}"
    try:
        critical_band, K_J = NFFT_FS_TO_CB_K_J[key]
    except KeyError:
        raise RuntimeError("n_fft {n_fft}, fs {fs} is not supported.")
    
    if key not in Tq:
        tq = absolute_threshold_of_hearing(n_fft, fs)
        Tq[key] = tq

        z = np.minimum(
            13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500)**2),
            critical_band
        )
        z[0] = z[1]
        Z[key] = z

        bark = np.arange(critical_band, dtype=np.float32)
        ones = np.ones((z.shape[0], critical_band), dtype=np.float32)
        cfb = np.where(
            np.expand_dims(z, 1) <= np.expand_dims((bark + 1), 0),
            ones,
            0.0
        )
        cfb = cfb - np.where(
            np.expand_dims(z, 1) <= np.expand_dims(bark, 0),
            ones,
            0.0
        )
        CFB[key] = cfb

        non_tonal_idx_global = np.where(
            cfb.astype(bool),
            np.expand_dims(np.arange(1, n_fft//2 + 2, dtype=np.float32), 1).repeat(critical_band, 1),
            np.ones_like(cfb)
        )
        non_tonal_idx_global = np.log(non_tonal_idx_global).sum(axis=0)
        non_tonal_idx_global = np.exp(non_tonal_idx_global / cfb.sum(axis=0)).astype(np.int64)
        NON_TONAL_IDX[key] = non_tonal_idx_global
    else:
        delta = fs / n_fft
        f = np.arange(0, fs // 2 + delta, delta, dtype=np.float32)
        tq = Tq[key]
        z = Z[key]
        cfb = CFB[key]
        non_tonal_idx_global = NON_TONAL_IDX[key]
    X = amp_to_spl(mag_spec, n_fft)                           # [B, n_fft//2+1, T]
    B, T = X.shape[0], X.shape[2]
    X = np.swapaxes(X, 1, 2).reshape(B * T, n_fft//2+1)
    X = np.ascontiguousarray(X)           # [BT, n_fft//2+1]

    # find local maximum
    is_local_maximum = local_maximum(X, K_J[0][0], K_J[-1][1])

    # find tonal maskers
    X_tm = np.ones_like(X) * (-np.inf) # [BT, n_fft//2+1]
    X_for_nm = X.copy()                        # [BT, n_fft//2+1]
    X_for_nm[:, 0] = -np.inf
    is_tonal = is_local_maximum                 # [BT, n_fft//2+1]
    for k_j in K_J:
        tonal_mask(X, X_tm, X_for_nm, is_tonal, *k_j)
    
    # find non-tonal(noise) maskers
    X_nm = non_tonal_mask(X_for_nm, cfb)      # [BT, critical_band]
    non_tonal_idx = np.expand_dims(non_tonal_idx_global, 0).repeat(X.shape[0], 0)
    non_tonal_idx = np.ascontiguousarray(non_tonal_idx)    # [BT, critical_band]

    # decimate -> X_tm, is_tonal, z: cpu / X_nm, non_tonal_idx: cuda
    X_tm, is_tonal, X_nm, non_tonal_idx, z = decimate(X_tm, is_tonal, X_nm, non_tonal_idx, tq, z)
    
    # find global masking threshold
    Tg = global_mask(X_tm, is_tonal, X_nm, non_tonal_idx, tq, non_tonal_idx_global, z)
    Tg = Tg.reshape(B, T, n_fft//2+1)
    Tg = np.swapaxes(Tg, 1, 2)
    
    return Tg


def amp_to_spl(mag_spec, n_fft):
    return 90.302 + 20.0 * np.log10(np.clip(mag_spec / n_fft, a_min=1.0e-7, a_max=None))


def spl_to_amp(spl_spec, n_fft):
    return 10.0 ** ((spl_spec - 90.302) / 20.0) * n_fft
