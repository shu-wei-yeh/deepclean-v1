import torch
import torch.nn as nn
import numpy as np
import time

# ---------- Compatibility helpers ----------
_HAS_TORCHFFT = hasattr(torch, "fft") and hasattr(torch.fft, "rfft")


def _hann_window(nperseg, device):
    try:
        return torch.hann_window(nperseg, device=device)
    except TypeError:
        return torch.hann_window(nperseg).to(device)


def _rfft(x, n=None):
    """
    Unified real FFT:
      - new torch: torch.fft.rfft(...) -> complex (..., F)
      - old torch: torch.rfft(..., onesided=True) -> real-imag stacked (..., F, 2)
    """
    if _HAS_TORCHFFT:
        return torch.fft.rfft(x, n=n)
    else:
        return torch.rfft(x, 1, onesided=True, normalized=False)


def _abs2(X):
    """|X|^2 for both representations."""
    if _HAS_TORCHFFT:
        return (X.real**2 + X.imag**2)
    else:
        return X[..., 0]**2 + X[..., 1]**2


def _grad_scale(x, scale):
    """
    Only scale gradients (straight-through trick):
      y = x*scale + x.detach()*(1 - scale)
    """
    return x * scale + x.detach() * (1.0 - scale)


# ---------- Welch scaling ----------
def _one_sided_scale(nperseg, fs, window):
    # Match scipy.signal.welch(scaling='density', return_onesided=True)
    U = (window**2).sum()
    return 1.0 / (fs * U)


# ---------- Welch PSD ----------
def _torch_welch(data, fs=1.0, nperseg=256, noverlap=None, average='mean', device='cpu'):
    # Flatten to (B, T)
    if len(data.shape) > 2:
        data = data.view(data.shape[0], -1)
    data = data.to(device)
    B, nsample = data.shape

    if noverlap is None:
        noverlap = nperseg // 2
    nstride = nperseg - noverlap
    if nstride <= 0:
        raise ValueError('overlap must be smaller than nperseg')

    nseg = 1 if nsample <= nperseg else int(
        np.floor((nsample - nperseg) / nstride)) + 1

    window = _hann_window(nperseg, device)
    scale = _one_sided_scale(nperseg, fs, window)

    if average == 'mean':
        acc = torch.zeros((B, nperseg // 2 + 1), device=device)
        for i in range(nseg):
            start = i * nstride
            seg = data[:, start:start + nperseg]
            if seg.shape[1] < nperseg:
                seg = nn.functional.pad(seg, (0, nperseg - seg.shape[1]))
            seg = seg * window
            X = _rfft(seg, n=nperseg)
            Pxx = _abs2(X)
            acc += Pxx
        acc /= nseg

    elif average == 'median':
        store = []
        for i in range(nseg):
            start = i * nstride
            seg = data[:, start:start + nperseg]
            if seg.shape[1] < nperseg:
                seg = nn.functional.pad(seg, (0, nperseg - seg.shape[1]))
            seg = seg * window
            X = _rfft(seg, n=nperseg)
            Pxx = _abs2(X).unsqueeze(0)  # (1, B, F)
            store.append(Pxx)
        acc = torch.median(torch.cat(store, dim=0), dim=0).values  # (B, F)

    else:
        raise ValueError('average must be "mean" or "median"')

    # one-sided doubling except DC/Nyquist
    if nperseg % 2 == 0:
        acc[:, 1:-1] *= 2.0
    else:
        acc[:, 1:] *= 2.0

    return acc * scale  # (B, F)


# ---------- Cross Welch ----------
def _torch_cross_welch(x, y, fs=1.0, nperseg=256, noverlap=None, average='mean', device='cpu'):
    # shape to (B, T)
    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1)
    if len(y.shape) > 2:
        y = y.view(y.shape[0], -1)
    x = x.to(device)
    y = y.to(device)
    B, nsample = x.shape

    if noverlap is None:
        noverlap = nperseg // 2
    nstride = nperseg - noverlap
    if nstride <= 0:
        raise ValueError('overlap must be smaller than nperseg')

    nseg = 1 if nsample <= nperseg else int(
        np.floor((nsample - nperseg) / nstride)) + 1

    window = _hann_window(nperseg, device)
    scale = _one_sided_scale(nperseg, fs, window)

    if average == 'mean':
        acc_r = torch.zeros((B, nperseg // 2 + 1), device=device)
        acc_i = torch.zeros((B, nperseg // 2 + 1), device=device)
        for i in range(nseg):
            start = i * nstride
            sx = x[:, start:start + nperseg]
            sy = y[:, start:start + nperseg]
            if sx.shape[1] < nperseg:
                pad = nperseg - sx.shape[1]
                sx = nn.functional.pad(sx, (0, pad))
                sy = nn.functional.pad(sy, (0, pad))
            sx = sx * window
            sy = sy * window
            X = _rfft(sx, n=nperseg)
            Y = _rfft(sy, n=nperseg)

            if _HAS_TORCHFFT:
                Sxy = X * torch.conj(Y)
                Sxy_r = Sxy.real
                Sxy_i = Sxy.imag
            else:
                Xr, Xi = X[..., 0], X[..., 1]
                Yr, Yi = Y[..., 0], Y[..., 1]
                # (a+jb)*(c-jd) = (ac+bd) + j(bc-ad)
                Sxy_r = Xr*Yr + Xi*Yi
                Sxy_i = Xi*Yr - Xr*Yi

            acc_r += Sxy_r
            acc_i += Sxy_i

        acc_r /= nseg
        acc_i /= nseg

    elif average == 'median':
        store_r, store_i = [], []
        for i in range(nseg):
            start = i * nstride
            sx = x[:, start:start + nperseg]
            sy = y[:, start:start + nperseg]
            if sx.shape[1] < nperseg:
                pad = nperseg - sx.shape[1]
                sx = nn.functional.pad(sx, (0, pad))
                sy = nn.functional.pad(sy, (0, pad))
            sx = sx * window
            sy = sy * window
            X = _rfft(sx, n=nperseg)
            Y = _rfft(sy, n=nperseg)

            if _HAS_TORCHFFT:
                Sxy = X * torch.conj(Y)
                Sxy_r = Sxy.real
                Sxy_i = Sxy.imag
            else:
                Xr, Xi = X[..., 0], X[..., 1]
                Yr, Yi = Y[..., 0], Y[..., 1]
                Sxy_r = Xr*Yr + Xi*Yi
                Sxy_i = Xi*Yr - Xr*Yi

            store_r.append(Sxy_r.unsqueeze(0))
            store_i.append(Sxy_i.unsqueeze(0))

        acc_r = torch.median(torch.cat(store_r, dim=0), dim=0).values
        acc_i = torch.median(torch.cat(store_i, dim=0), dim=0).values

    else:
        raise ValueError('average must be "mean" or "median"')

    # one-sided doubling except DC/Nyquist
    if nperseg % 2 == 0:
        acc_r[:, 1:-1] *= 2.0
        acc_i[:, 1:-1] *= 2.0
    else:
        acc_r[:, 1:] *= 2.0
        acc_i[:, 1:] *= 2.0

    acc_r = acc_r * scale
    acc_i = acc_i * scale
    return acc_r, acc_i  # (B, F)


# ---------- Coherence ----------
def compute_coherence(x, y, fs, nperseg, noverlap=None, device='cpu', average='mean'):
    x = x.to(device)
    y = y.to(device)
    Sxx = _torch_welch(x, fs, nperseg, noverlap, average, device)
    Syy = _torch_welch(y, fs, nperseg, noverlap, average, device)
    Sxy_r, Sxy_i = _torch_cross_welch(
        x, y, fs, nperseg, noverlap, average, device)
    Sxy_mag2 = Sxy_r**2 + Sxy_i**2
    return Sxy_mag2 / (Sxx * Syy + 1e-12)


# ---------- Batched Welch for witness channel chunks ----------
def _torch_welch_channels(data, fs=1.0, nperseg=256, noverlap=None,
                          average='mean', device='cpu'):
    """
    Welch PSD for multi-channel witness data.

    Parameters
    ----------
    data : torch.Tensor
        Shape (B, C, T)

    Returns
    -------
    psd : torch.Tensor
        Shape (B, C, F)
    """

    if len(data.shape) != 3:
        raise ValueError("data must have shape (B, C, T)")

    data = data.to(device)
    B, C, nsample = data.shape

    if noverlap is None:
        noverlap = nperseg // 2

    nstride = nperseg - noverlap
    if nstride <= 0:
        raise ValueError("overlap must be smaller than nperseg")

    nseg = 1 if nsample <= nperseg else int(
        np.floor((nsample - nperseg) / nstride)
    ) + 1

    window = _hann_window(nperseg, device)
    scale = _one_sided_scale(nperseg, fs, window)
    nfreq = nperseg // 2 + 1

    if average == 'mean':
        acc = torch.zeros((B, C, nfreq), device=device)

        for i in range(nseg):
            start = i * nstride
            seg = data[:, :, start:start + nperseg]

            if seg.shape[2] < nperseg:
                pad = nperseg - seg.shape[2]
                seg = nn.functional.pad(seg, (0, pad))

            seg = seg * window.view(1, 1, -1)

            # Old torch.rfft is safer with 2D input.
            seg_flat = seg.contiguous().view(B * C, nperseg)
            X = _rfft(seg_flat, n=nperseg)
            Pxx = _abs2(X).view(B, C, nfreq)

            acc += Pxx

        acc /= nseg

    elif average == 'median':
        store = []

        for i in range(nseg):
            start = i * nstride
            seg = data[:, :, start:start + nperseg]

            if seg.shape[2] < nperseg:
                pad = nperseg - seg.shape[2]
                seg = nn.functional.pad(seg, (0, pad))

            seg = seg * window.view(1, 1, -1)

            seg_flat = seg.contiguous().view(B * C, nperseg)
            X = _rfft(seg_flat, n=nperseg)
            Pxx = _abs2(X).view(B, C, nfreq)

            store.append(Pxx.unsqueeze(0))

        acc = torch.median(torch.cat(store, dim=0), dim=0).values

    else:
        raise ValueError('average must be "mean" or "median"')

    # One-sided doubling except DC/Nyquist.
    if nperseg % 2 == 0:
        acc[:, :, 1:-1] *= 2.0
    else:
        acc[:, :, 1:] *= 2.0

    return acc * scale


def _torch_cross_welch_x_channels(x, y, fs=1.0, nperseg=256, noverlap=None,
                                  average='mean', device='cpu'):
    """
    Cross Welch between one strain-like time series x and a chunk of
    witness channels y.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, T)

    y : torch.Tensor
        Shape (B, C, T)

    Returns
    -------
    Sxy_r, Sxy_i : torch.Tensor
        Real and imaginary parts of cross spectrum, shape (B, C, F)
    """

    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1)

    if len(y.shape) != 3:
        raise ValueError("y must have shape (B, C, T)")

    x = x.to(device)
    y = y.to(device)

    B, nsample = x.shape
    By, C, ysample = y.shape

    if By != B:
        raise ValueError("x and y must have the same batch size")

    if noverlap is None:
        noverlap = nperseg // 2

    nstride = nperseg - noverlap
    if nstride <= 0:
        raise ValueError("overlap must be smaller than nperseg")

    nseg = 1 if nsample <= nperseg else int(
        np.floor((nsample - nperseg) / nstride)
    ) + 1

    window = _hann_window(nperseg, device)
    scale = _one_sided_scale(nperseg, fs, window)
    nfreq = nperseg // 2 + 1

    if average == 'mean':
        acc_r = torch.zeros((B, C, nfreq), device=device)
        acc_i = torch.zeros((B, C, nfreq), device=device)

        for i in range(nseg):
            start = i * nstride

            sx = x[:, start:start + nperseg]
            sy = y[:, :, start:start + nperseg]

            if sx.shape[1] < nperseg:
                pad = nperseg - sx.shape[1]
                sx = nn.functional.pad(sx, (0, pad))
                sy = nn.functional.pad(sy, (0, pad))

            sx = sx * window.view(1, -1)
            sy = sy * window.view(1, 1, -1)

            X = _rfft(sx, n=nperseg)

            sy_flat = sy.contiguous().view(B * C, nperseg)
            Y = _rfft(sy_flat, n=nperseg)

            if _HAS_TORCHFFT:
                X = X.view(B, 1, nfreq)
                Y = Y.view(B, C, nfreq)

                Sxy = X * torch.conj(Y)
                Sxy_r = Sxy.real
                Sxy_i = Sxy.imag

            else:
                Xr = X[..., 0].view(B, 1, nfreq)
                Xi = X[..., 1].view(B, 1, nfreq)

                Y = Y.view(B, C, nfreq, 2)
                Yr = Y[..., 0]
                Yi = Y[..., 1]

                # (a+jb)*(c-jd) = (ac+bd) + j(bc-ad)
                Sxy_r = Xr * Yr + Xi * Yi
                Sxy_i = Xi * Yr - Xr * Yi

            acc_r += Sxy_r
            acc_i += Sxy_i

        acc_r /= nseg
        acc_i /= nseg

    elif average == 'median':
        store_r = []
        store_i = []

        for i in range(nseg):
            start = i * nstride

            sx = x[:, start:start + nperseg]
            sy = y[:, :, start:start + nperseg]

            if sx.shape[1] < nperseg:
                pad = nperseg - sx.shape[1]
                sx = nn.functional.pad(sx, (0, pad))
                sy = nn.functional.pad(sy, (0, pad))

            sx = sx * window.view(1, -1)
            sy = sy * window.view(1, 1, -1)

            X = _rfft(sx, n=nperseg)

            sy_flat = sy.contiguous().view(B * C, nperseg)
            Y = _rfft(sy_flat, n=nperseg)

            if _HAS_TORCHFFT:
                X = X.view(B, 1, nfreq)
                Y = Y.view(B, C, nfreq)

                Sxy = X * torch.conj(Y)
                Sxy_r = Sxy.real
                Sxy_i = Sxy.imag

            else:
                Xr = X[..., 0].view(B, 1, nfreq)
                Xi = X[..., 1].view(B, 1, nfreq)

                Y = Y.view(B, C, nfreq, 2)
                Yr = Y[..., 0]
                Yi = Y[..., 1]

                Sxy_r = Xr * Yr + Xi * Yi
                Sxy_i = Xi * Yr - Xr * Yi

            store_r.append(Sxy_r.unsqueeze(0))
            store_i.append(Sxy_i.unsqueeze(0))

        acc_r = torch.median(torch.cat(store_r, dim=0), dim=0).values
        acc_i = torch.median(torch.cat(store_i, dim=0), dim=0).values

    else:
        raise ValueError('average must be "mean" or "median"')

    # One-sided doubling except DC/Nyquist.
    if nperseg % 2 == 0:
        acc_r[:, :, 1:-1] *= 2.0
        acc_i[:, :, 1:-1] *= 2.0
    else:
        acc_r[:, :, 1:] *= 2.0
        acc_i[:, :, 1:] *= 2.0

    acc_r = acc_r * scale
    acc_i = acc_i * scale

    return acc_r, acc_i


# ---------- Losses ----------
class MSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError('`reduction` must be "mean" or "sum"')
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        loss = (target - pred) ** 2
        loss = torch.mean(loss, 1)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / len(pred)
        else:
            loss = torch.sum(loss)
        return loss


class PSDLoss(nn.Module):
    def __init__(self, fs=1.0, fl=20., fh=500., fftlength=1., overlap=None,
                 asd=True, average='mean', reduction='mean', device='cpu'):
        super().__init__()
        if isinstance(fl, (int, float)):
            fl = (fl,)
        if isinstance(fh, (int, float)):
            fh = (fh,)
        if reduction not in ('mean', 'sum'):
            raise ValueError('`reduction` must be "mean" or "sum"')
        self.reduction = reduction
        self.fs = fs
        self.average = average
        self.device = device
        self.asd = asd

        nperseg = int(fftlength * self.fs)
        noverlap = int(overlap * self.fs) if overlap is not None else None
        self.welch = lambda x: _torch_welch(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap, average=average, device=device)

        freq = torch.linspace(0., fs/2., nperseg//2 + 1, device=device)
        self.dfreq = freq[1] - freq[0]
        # uint8 for old torch
        self.mask = torch.zeros(
            nperseg//2 + 1, dtype=torch.uint8, device=device)
        self.scale = 0.0
        for l, h in zip(fl, fh):
            band = ((l < freq) & (freq < h)).to(torch.uint8)
            self.mask = torch.max(self.mask, band)  # OR
            self.scale += (h - l)

    def forward(self, pred, target):
        psd_res = self.welch(target - pred)      # (B, F)
        psd_target = self.welch(target)          # (B, F)
        # zero out-of-band
        mask_eq0 = (self.mask == 0)
        psd_res[:, mask_eq0] = 0.0

        psd_ratio = psd_res / (psd_target + 1e-13)
        asd_ratio = torch.sqrt(psd_ratio)

        if self.asd:
            loss = torch.sum(asd_ratio, 1) * self.dfreq / self.scale
        else:
            loss = torch.sum(psd_ratio, 1) * self.dfreq / self.scale

        if self.reduction == 'mean':
            loss = torch.sum(loss) / len(psd_res)
        else:
            loss = torch.sum(loss)
        return loss


class CoherenceLoss(nn.Module):
    """
    Band-limited residual-coherence loss.

    residual = target - pred

    The loss is evaluated only where the original target-witness coherence is
    meaningful.  Witnesses are processed in chunks to reduce peak GPU memory.

    Important defaults
    ------------------
    target_coh_gate = 0.05
        Frequencies with target-witness coherence below this value do not
        contribute to the loss.

    channel_coh_threshold = 0.03
        Channels whose mean target coherence is below this level receive zero
        channel weight.

    fallback_to_uniform_channels = False
        If no channel has meaningful target coherence, this loss returns zero
        rather than forcing the model to optimize unrelated witnesses.
    """

    def __init__(
        self,
        fs,
        fl,
        fh,
        fftlength=1.0,
        overlap=None,
        reduction='mean',
        device='cpu',
        average='mean',
        coh_floor=1e-3,
        target_coh_gate=0.05,
        weight_by_target_coh=True,
        use_log=True,
        channel_coh_threshold=0.03,
        channel_weight_power=1.0,
        min_channel_weight_sum=1e-8,
        fallback_to_uniform_channels=False,
        channel_chunk_size=8,
    ):
        super().__init__()

        if reduction not in ('mean', 'sum'):
            raise ValueError('`reduction` must be "mean" or "sum"')

        self.fs = fs
        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap is not None else None
        self.reduction = reduction
        self.device = device
        self.average = average

        self.coh_floor = float(coh_floor)
        self.target_coh_gate = float(target_coh_gate)
        self.weight_by_target_coh = bool(weight_by_target_coh)
        self.use_log = bool(use_log)

        self.channel_coh_threshold = float(channel_coh_threshold)
        self.channel_weight_power = float(channel_weight_power)
        self.min_channel_weight_sum = float(min_channel_weight_sum)
        self.fallback_to_uniform_channels = bool(fallback_to_uniform_channels)

        self.channel_chunk_size = int(channel_chunk_size)
        if self.channel_chunk_size <= 0:
            raise ValueError("channel_chunk_size must be positive")

        freq = torch.linspace(
            0.0, fs / 2.0, self.nperseg // 2 + 1, device=device
        )

        if isinstance(fl, (int, float)):
            fl = [fl]
        if isinstance(fh, (int, float)):
            fh = [fh]

        freq_mask = torch.zeros_like(freq, dtype=torch.uint8, device=device)
        for l, h in zip(fl, fh):
            band = ((freq >= l) & (freq <= h)).to(torch.uint8)
            freq_mask = torch.max(freq_mask, band)

        self.register_buffer("freq_mask", freq_mask)
        self.register_buffer("freq_mask_float", freq_mask.float())

        n_band_bins = torch.sum(freq_mask.float())
        self.register_buffer(
            "n_band_bins",
            torch.clamp(n_band_bins, min=1.0),
        )

    def _welch_1d(self, x):
        return _torch_welch(
            x,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def _welch_channels(self, x):
        return _torch_welch_channels(
            x,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def _cross_x_channels(self, x, y):
        return _torch_cross_welch_x_channels(
            x,
            y,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def forward(self, pred, target, witness):
        residual = target - pred

        if witness.ndim != 3:
            raise ValueError("witness must have shape (B, C, T)")

        B, C, _ = witness.shape

        if C == 0:
            return torch.sum(pred * 0.0)

        band_mask = self.freq_mask_float.view(1, 1, -1)

        # residual PSD carries gradients; target PSD is reference-only
        Srr = self._welch_1d(residual).view(B, 1, -1)
        with torch.no_grad():
            Stt = self._welch_1d(target).view(B, 1, -1)

        weighted_num = torch.zeros(B, device=pred.device, dtype=pred.dtype)
        weighted_den = torch.zeros(B, device=pred.device, dtype=pred.dtype)

        uniform_num = torch.zeros(B, device=pred.device, dtype=pred.dtype)
        uniform_den = torch.zeros(B, device=pred.device, dtype=pred.dtype)

        for start in range(0, C, self.channel_chunk_size):
            end = min(start + self.channel_chunk_size, C)
            w_chunk = witness[:, start:end, :]

            with torch.no_grad():
                Sww = self._welch_channels(w_chunk)
                Stw_r, Stw_i = self._cross_x_channels(target, w_chunk)
                Stw_mag2 = Stw_r ** 2 + Stw_i ** 2
                coh_tgt_ref = (
                    Stw_mag2 / (Stt * Sww + 1e-12)
                ).detach()

            Srw_r, Srw_i = self._cross_x_channels(residual, w_chunk)
            Srw_mag2 = Srw_r ** 2 + Srw_i ** 2
            coh_res = Srw_mag2 / (Srr * Sww.detach() + 1e-12)

            ratio = coh_res / torch.clamp(
                coh_tgt_ref, min=self.coh_floor
            )

            if self.use_log:
                ratio = torch.log1p(ratio)

            # Only train on frequencies that had meaningful original coupling.
            valid_freq = (
                (coh_tgt_ref >= self.target_coh_gate).type_as(ratio)
                * band_mask
            )

            if self.weight_by_target_coh:
                freq_weights = coh_tgt_ref * valid_freq
            else:
                freq_weights = valid_freq

            freq_den = torch.sum(freq_weights, dim=2)

            # Safe division. Channels with no valid frequencies get loss=0.
            loss_chunk = torch.sum(
                ratio * freq_weights, dim=2
            ) / (freq_den + 1e-12)

            has_valid_freq = (freq_den > 0).type_as(loss_chunk)
            loss_chunk = loss_chunk * has_valid_freq

            # Channel strength uses target coherence over the target band.
            channel_strength = torch.sum(
                coh_tgt_ref * band_mask, dim=2
            ) / (self.n_band_bins + 1e-12)

            channel_weight = torch.clamp(
                channel_strength - self.channel_coh_threshold,
                min=0.0,
            ) ** self.channel_weight_power

            channel_weight = (
                channel_weight.detach() * has_valid_freq.detach()
            )

            weighted_num = weighted_num + torch.sum(
                loss_chunk * channel_weight, dim=1
            )
            weighted_den = weighted_den + torch.sum(
                channel_weight, dim=1
            )

            uniform_num = uniform_num + torch.sum(
                loss_chunk, dim=1
            )
            uniform_den = uniform_den + torch.sum(
                has_valid_freq, dim=1
            )

        weighted_loss = weighted_num / (weighted_den + 1e-12)
        uniform_loss = uniform_num / (uniform_den + 1e-12)

        if self.fallback_to_uniform_channels:
            use_uniform = (
                weighted_den <= self.min_channel_weight_sum
            ).type_as(weighted_loss)
            loss = (
                weighted_loss * (1.0 - use_uniform)
                + uniform_loss * use_uniform
            )
        else:
            # Preferred behavior: if no physically meaningful witness exists,
            # make the coherence term zero for that batch item.
            has_weight = (
                weighted_den > self.min_channel_weight_sum
            ).type_as(weighted_loss)
            loss = weighted_loss * has_weight

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class TransferFunctionLoss(nn.Module):
    """
    Band-limited transfer-function loss with coherence gating.

    The TF term is only evaluated where the original target-witness coherence
    is meaningful.  This avoids strongly penalizing frequencies where the
    transfer-function estimate is not well supported.

    By default gradient boosting is disabled.  Control the strength mainly with
    tf_weight in CompositePSDLoss.
    """

    def __init__(
        self,
        fs,
        fl,
        fh,
        fftlength=1.0,
        overlap=None,
        reduction='mean',
        device='cpu',
        average='mean',
        nonlinearity='log1p',
        mode='ratio',
        syy_floor_factor=1e-6,
        tf_floor_factor=1e-6,
        target_coh_gate=0.05,
        channel_coh_threshold=0.03,
        channel_weight_power=1.0,
        fallback_to_uniform_channels=False,
        channel_chunk_size=8,
        center_weighting=False,

        # Backward-compatible gradient scaling options.
        grad_boost=1.0,
        auto_grad_boost=False,
        max_grad_boost=100.0,
    ):
        super().__init__()

        if reduction not in ('mean', 'sum'):
            raise ValueError('`reduction` must be "mean" or "sum"')
        if mode not in ('ratio', 'abs'):
            raise ValueError("`mode` must be either 'ratio' or 'abs'")

        self.fs = fs
        self.device = device
        self.average = average
        self.reduction = reduction
        self.nonlinearity = nonlinearity
        self.mode = mode

        self.syy_floor_factor = float(syy_floor_factor)
        self.tf_floor_factor = float(tf_floor_factor)
        self.target_coh_gate = float(target_coh_gate)
        self.channel_coh_threshold = float(channel_coh_threshold)
        self.channel_weight_power = float(channel_weight_power)
        self.fallback_to_uniform_channels = bool(
            fallback_to_uniform_channels
        )

        self.channel_chunk_size = int(channel_chunk_size)
        if self.channel_chunk_size <= 0:
            raise ValueError("channel_chunk_size must be positive")

        self.center_weighting = bool(center_weighting)

        # Gradient boost is intentionally conservative/off by default.
        self.grad_boost = float(grad_boost)
        self.auto_grad_boost = bool(auto_grad_boost)
        self.max_grad_boost = float(max_grad_boost)
        self.grad_target = 1.0
        self.ema_beta = 0.9

        self.register_buffer(
            "gb_ema",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "gb_inited",
            torch.tensor(0, dtype=torch.uint8, device=device),
        )

        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap is not None else None

        freqs = torch.linspace(
            0.0, fs / 2.0, self.nperseg // 2 + 1, device=device
        )

        if isinstance(fl, (int, float)):
            fl = [fl]
        if isinstance(fh, (int, float)):
            fh = [fh]

        band_mask = torch.zeros_like(freqs, dtype=torch.uint8)
        base_weight = torch.zeros_like(freqs)

        for l, h in zip(fl, fh):
            band = ((freqs >= l) & (freqs <= h)).to(torch.uint8)
            band_mask = torch.max(band_mask, band)

            if self.center_weighting:
                center = (l + h) / 2.0
                width = max(h - l, 1e-6)
                sigma = width / 4.0
                gaussian = torch.exp(
                    -0.5 * ((freqs - center) / sigma) ** 2
                )
                base_weight = torch.max(base_weight, gaussian)
            else:
                base_weight = torch.max(
                    base_weight, band.float()
                )

        base_weight = base_weight * band_mask.float()
        if torch.sum(base_weight) <= 0:
            base_weight = band_mask.float()

        self.register_buffer("freqs", freqs)
        self.register_buffer("band_mask", band_mask)
        self.register_buffer("band_mask_float", band_mask.float())
        self.register_buffer("base_weight", base_weight)

        n_band_bins = torch.sum(band_mask.float())
        self.register_buffer(
            "n_band_bins",
            torch.clamp(n_band_bins, min=1.0),
        )

    def _welch_1d(self, x):
        return _torch_welch(
            x,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def _welch_channels(self, x):
        return _torch_welch_channels(
            x,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def _cross_x_channels(self, x, y):
        return _torch_cross_welch_x_channels(
            x,
            y,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def _apply_nonlinearity(self, x):
        if self.nonlinearity == 'log1p':
            return torch.log1p(x)
        if self.nonlinearity == 'sqrt':
            return torch.sqrt(x + 1e-12)
        if self.nonlinearity in ('none', None):
            return x
        raise ValueError(
            "`nonlinearity` must be 'log1p', 'sqrt', or 'none'"
        )

    def _compute_grad_scale(self, x):
        if not self.auto_grad_boost:
            return self.grad_boost

        with torch.no_grad():
            cur_med = x.detach().median().clamp(min=1e-12)

            if self.gb_inited.item() == 0:
                self.gb_ema.copy_(cur_med.float())
                self.gb_inited.fill_(1)

            self.gb_ema.mul_(self.ema_beta).add_(
                (1.0 - self.ema_beta) * cur_med.float()
            )

            scale = (
                self.grad_target * self.grad_boost
            ) / (float(self.gb_ema.item()) + 1e-12)

            return max(1.0, min(scale, self.max_grad_boost))

    def forward(self, pred, target, witness):
        residual = target - pred

        if witness.ndim != 3:
            raise ValueError("witness must have shape (B, C, T)")

        B, C, _ = witness.shape
        if C == 0:
            return torch.sum(pred * 0.0)

        band_mask = self.band_mask_float.view(1, 1, -1)
        base_weight = self.base_weight.view(1, 1, -1)

        # Target PSD is a reference quantity.
        with torch.no_grad():
            Stt = self._welch_1d(target).view(B, 1, -1)

        weighted_num = torch.zeros(B, device=pred.device, dtype=pred.dtype)
        weighted_den = torch.zeros(B, device=pred.device, dtype=pred.dtype)

        uniform_num = torch.zeros(B, device=pred.device, dtype=pred.dtype)
        uniform_den = torch.zeros(B, device=pred.device, dtype=pred.dtype)

        for start in range(0, C, self.channel_chunk_size):
            end = min(start + self.channel_chunk_size, C)
            w_chunk = witness[:, start:end, :]

            # References: no gradient needed.
            with torch.no_grad():
                Sww = self._welch_channels(w_chunk)
                Stw_r, Stw_i = self._cross_x_channels(
                    target, w_chunk
                )
                Stw_mag2 = Stw_r ** 2 + Stw_i ** 2

                coh_tgt_ref = (
                    Stw_mag2 / (Stt * Sww + 1e-12)
                ).detach()

                # Adaptive Sww floor based on band mean.
                sww_band_mean = torch.sum(
                    Sww * band_mask, dim=2, keepdim=True
                ) / (self.n_band_bins + 1e-12)

                syy_floor = (
                    self.syy_floor_factor * sww_band_mean
                    + 1e-12
                ).type_as(Sww)

                Sww_safe = torch.maximum(
                    Sww, syy_floor.expand_as(Sww)
                )

                Htgt2 = Stw_mag2 / (Sww_safe ** 2 + 1e-24)
                Htgt2_ref = Htgt2.detach()

            # Residual-witness cross spectrum keeps gradient through residual.
            Srw_r, Srw_i = self._cross_x_channels(
                residual, w_chunk
            )
            Srw_mag2 = Srw_r ** 2 + Srw_i ** 2

            Hres2 = Srw_mag2 / (
                Sww_safe.detach() ** 2 + 1e-24
            )

            if self.mode == 'ratio':
                h_tgt_band = torch.sum(
                    Htgt2_ref * band_mask, dim=2, keepdim=True
                ) / (self.n_band_bins + 1e-12)

                tf_floor = (
                    self.tf_floor_factor * h_tgt_band
                    + 1e-12
                ).type_as(Htgt2_ref)

                Htgt2_safe = torch.maximum(
                    Htgt2_ref,
                    tf_floor.expand_as(Htgt2_ref),
                )

                x = Hres2 / Htgt2_safe
            else:
                x = Hres2

            x = self._apply_nonlinearity(x)

            # Only use TF where original coherence supports a real coupling.
            valid_freq = (
                (coh_tgt_ref >= self.target_coh_gate).type_as(x)
                * band_mask
            )

            # Share the same physics idea as CoherenceLoss:
            # high original coherence gets higher frequency weight.
            freq_weights = (
                base_weight
                * coh_tgt_ref
                * valid_freq
            )

            freq_den = torch.sum(freq_weights, dim=2)
            loss_chunk = torch.sum(
                x * freq_weights, dim=2
            ) / (freq_den + 1e-12)

            has_valid_freq = (freq_den > 0).type_as(loss_chunk)
            loss_chunk = loss_chunk * has_valid_freq

            # Optional gradient-only scaling; default is exactly 1.
            scale = self._compute_grad_scale(loss_chunk)
            if scale != 1.0:
                loss_chunk = _grad_scale(loss_chunk, scale)

            channel_strength = torch.sum(
                coh_tgt_ref * band_mask, dim=2
            ) / (self.n_band_bins + 1e-12)

            channel_weight = torch.clamp(
                channel_strength - self.channel_coh_threshold,
                min=0.0,
            ) ** self.channel_weight_power

            channel_weight = (
                channel_weight.detach() * has_valid_freq.detach()
            )

            weighted_num = weighted_num + torch.sum(
                loss_chunk * channel_weight, dim=1
            )
            weighted_den = weighted_den + torch.sum(
                channel_weight, dim=1
            )

            uniform_num = uniform_num + torch.sum(
                loss_chunk, dim=1
            )
            uniform_den = uniform_den + torch.sum(
                has_valid_freq, dim=1
            )

        weighted_loss = weighted_num / (weighted_den + 1e-12)
        uniform_loss = uniform_num / (uniform_den + 1e-12)

        if self.fallback_to_uniform_channels:
            use_uniform = (weighted_den <= 1e-8).type_as(weighted_loss)
            loss = (
                weighted_loss * (1.0 - use_uniform)
                + uniform_loss * use_uniform
            )
        else:
            has_weight = (weighted_den > 1e-8).type_as(weighted_loss)
            loss = weighted_loss * has_weight

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class CompositePSDLoss(nn.Module):
    """
    Composite DeepClean loss.

    IMPORTANT:
    The weights are independent coefficients. They are NOT normalized and
    do NOT need to sum to 1.

    Recommended baseline for the user's current comparison:
        PSD = 1.0
        MSE = 0.0
        COH = 0.0
        TF  = 0.0

    Suggested additive studies:
        PSD=1.0, COH=0.03
        PSD=1.0, COH=0.05
        PSD=1.0, COH=0.07
        PSD=1.0, TF=0.001
        PSD=1.0, COH=0.05, TF=0.001

    Timing
    ------
    Set enable_timing=True to measure forward-computation time of each loss.
    GPU timing uses torch.cuda.synchronize() around each timed component.
    This is accurate for profiling but adds synchronization overhead, so use
    it for benchmark runs rather than every production run.
    """

    def __init__(
        self,
        fs,
        fl,
        fh,
        fftlength=1.0,
        overlap=None,
        reduction='mean',
        device='cpu',

        # Preserve the user's PSD-only historical baseline.
        psd_weight=1.0,
        mse_weight=0.0,
        coh_weight=0.0,
        tf_weight=0.0,

        average='mean',
        nonlinearity='log1p',
        asd=True,

        # Coherence/TF physics controls.
        target_coh_gate=0.05,
        channel_coh_threshold=0.03,
        channel_chunk_size=8,

        # Timing controls.
        enable_timing=False,
        timing_warmup=5,
    ):
        super().__init__()

        if any(
            float(w) < 0.0
            for w in (
                psd_weight,
                mse_weight,
                coh_weight,
                tf_weight,
            )
        ):
            raise ValueError("Loss weights must be non-negative")

        if (
            float(psd_weight)
            + float(mse_weight)
            + float(coh_weight)
            + float(tf_weight)
        ) <= 0:
            raise ValueError("At least one loss weight must be > 0")

        # Deliberately DO NOT check that weights sum to 1.
        self.reduction = reduction
        self.psd_weight = float(psd_weight)
        self.mse_weight = float(mse_weight)
        self.coh_weight = float(coh_weight)
        self.tf_weight = float(tf_weight)
        self.device = device
        self.asd = asd

        self.mse_loss = MSELoss(reduction=reduction)

        self.psd_loss = PSDLoss(
            fs=fs,
            fl=fl,
            fh=fh,
            fftlength=fftlength,
            overlap=overlap,
            asd=asd,
            average=average,
            reduction=reduction,
            device=device,
        )

        self.coh_loss = CoherenceLoss(
            fs=fs,
            fl=fl,
            fh=fh,
            fftlength=fftlength,
            overlap=overlap,
            reduction=reduction,
            device=device,
            average=average,
            target_coh_gate=target_coh_gate,
            channel_coh_threshold=channel_coh_threshold,
            fallback_to_uniform_channels=False,
            channel_chunk_size=channel_chunk_size,
        )

        self.tf_loss = TransferFunctionLoss(
            fs=fs,
            fl=fl,
            fh=fh,
            fftlength=fftlength,
            overlap=overlap,
            reduction=reduction,
            device=device,
            average=average,
            nonlinearity=nonlinearity,
            target_coh_gate=target_coh_gate,
            channel_coh_threshold=channel_coh_threshold,
            fallback_to_uniform_channels=False,
            channel_chunk_size=channel_chunk_size,
            grad_boost=1.0,
            auto_grad_boost=False,
        )

        self.latest_loss_values = {
            'mse': 0.0,
            'psd': 0.0,
            'coh': 0.0,
            'tf': 0.0,
            'total': 0.0,
        }

        self.latest_weighted_loss_values = {
            'mse': 0.0,
            'psd': 0.0,
            'coh': 0.0,
            'tf': 0.0,
            'total': 0.0,
        }

        self.enable_timing = bool(enable_timing)
        self.timing_warmup = max(0, int(timing_warmup))
        self._forward_calls = 0

        self.latest_loss_timings = {
            'mse': 0.0,
            'psd': 0.0,
            'coh': 0.0,
            'tf': 0.0,
            'criterion_total': 0.0,
        }

        self._timing_total_seconds = {
            'mse': 0.0,
            'psd': 0.0,
            'coh': 0.0,
            'tf': 0.0,
            'criterion_total': 0.0,
        }

        self._timing_calls = {
            'mse': 0,
            'psd': 0,
            'coh': 0,
            'tf': 0,
            'criterion_total': 0,
        }

    def _sync_if_needed(self, ref_tensor):
        if (
            self.enable_timing
            and isinstance(ref_tensor, torch.Tensor)
            and ref_tensor.is_cuda
        ):
            torch.cuda.synchronize(ref_tensor.device)

    def _timed_call(self, name, fn, ref_tensor):
        if not self.enable_timing:
            return fn(), 0.0

        self._sync_if_needed(ref_tensor)
        t0 = time.perf_counter()
        value = fn()
        self._sync_if_needed(ref_tensor)
        elapsed = time.perf_counter() - t0

        return value, elapsed

    def reset_timing_stats(self):
        self._forward_calls = 0
        for key in self.latest_loss_timings:
            self.latest_loss_timings[key] = 0.0
            self._timing_total_seconds[key] = 0.0
            self._timing_calls[key] = 0

    def get_timing_summary(self):
        """
        Returns a dictionary:
            {
              'psd': {'calls': ..., 'total_s': ..., 'mean_ms': ...},
              ...
            }
        """
        summary = {}
        for key in self._timing_total_seconds:
            calls = self._timing_calls[key]
            total_s = self._timing_total_seconds[key]
            summary[key] = {
                'calls': calls,
                'total_s': total_s,
                'mean_ms': (
                    1000.0 * total_s / calls
                    if calls > 0 else 0.0
                ),
            }
        return summary

    def format_timing_summary(self):
        summary = self.get_timing_summary()
        lines = [
            "Loss forward timing "
            "(warm-up calls excluded):"
        ]
        for key in ('psd', 'mse', 'coh', 'tf', 'criterion_total'):
            s = summary[key]
            lines.append(
                f"  {key:15s}: "
                f"{s['mean_ms']:10.3f} ms/call  "
                f"{s['total_s']:10.3f} s total  "
                f"({s['calls']} calls)"
            )
        return "\n".join(lines)

    def _record_timing(self, key, elapsed, record):
        self.latest_loss_timings[key] = float(elapsed)
        if record:
            self._timing_total_seconds[key] += float(elapsed)
            self._timing_calls[key] += 1

    def forward(
        self,
        pred,
        target,
        witness,
        return_dict=False,
    ):
        self._forward_calls += 1
        record_timing = (
            self.enable_timing
            and self._forward_calls > self.timing_warmup
        )

        # Reset every call so inactive terms never show stale values.
        raw = {
            'mse': 0.0,
            'psd': 0.0,
            'coh': 0.0,
            'tf': 0.0,
        }
        weighted = {
            'mse': 0.0,
            'psd': 0.0,
            'coh': 0.0,
            'tf': 0.0,
        }
        for key in self.latest_loss_timings:
            self.latest_loss_timings[key] = 0.0

        self._sync_if_needed(pred)
        total_t0 = time.perf_counter() if self.enable_timing else None

        # Tensor-valued zero keeps graph/device/dtype consistent.
        loss = torch.sum(pred * 0.0)

        if self.psd_weight > 0:
            v, elapsed = self._timed_call(
                'psd',
                lambda: self.psd_loss(pred, target),
                pred,
            )
            loss = loss + self.psd_weight * v
            raw['psd'] = float(v.detach().item())
            weighted['psd'] = self.psd_weight * raw['psd']
            self._record_timing('psd', elapsed, record_timing)

        if self.mse_weight > 0:
            v, elapsed = self._timed_call(
                'mse',
                lambda: self.mse_loss(pred, target),
                pred,
            )
            loss = loss + self.mse_weight * v
            raw['mse'] = float(v.detach().item())
            weighted['mse'] = self.mse_weight * raw['mse']
            self._record_timing('mse', elapsed, record_timing)

        if self.coh_weight > 0:
            v, elapsed = self._timed_call(
                'coh',
                lambda: self.coh_loss(pred, target, witness),
                pred,
            )
            loss = loss + self.coh_weight * v
            raw['coh'] = float(v.detach().item())
            weighted['coh'] = self.coh_weight * raw['coh']
            self._record_timing('coh', elapsed, record_timing)

        if self.tf_weight > 0:
            v, elapsed = self._timed_call(
                'tf',
                lambda: self.tf_loss(pred, target, witness),
                pred,
            )
            loss = loss + self.tf_weight * v
            raw['tf'] = float(v.detach().item())
            weighted['tf'] = self.tf_weight * raw['tf']
            self._record_timing('tf', elapsed, record_timing)

        if self.enable_timing:
            self._sync_if_needed(pred)
            total_elapsed = time.perf_counter() - total_t0
            self._record_timing(
                'criterion_total',
                total_elapsed,
                record_timing,
            )

        total_value = float(loss.detach().item())
        self.latest_loss_values = {
            **raw,
            'total': total_value,
        }
        self.latest_weighted_loss_values = {
            **weighted,
            'total': total_value,
        }

        if return_dict:
            return loss, {
                'raw': dict(self.latest_loss_values),
                'weighted': dict(self.latest_weighted_loss_values),
                'timing_s': dict(self.latest_loss_timings),
            }

        return loss
