import torch
import torch.nn as nn
import numpy as np

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
                 asd=False, average='mean', reduction='mean', device='cpu'):
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
    Memory-optimized band-limited residual coherence loss.

    Key points
    ----------
    1. All witness channels are still used.
    2. Witness channels are processed in chunks to reduce peak memory.
    3. residual PSD is computed only once.
    4. target PSD is computed only once under no_grad.
    5. target-witness coherence is computed under no_grad.
    6. channel losses are accumulated in streaming form,
       without stacking all channel losses and weights.

    Loss idea
    ---------
    residual = target - pred

    Minimize:

        coherence(residual, witness)
        ----------------------------
        coherence(target, witness)

    inside the selected frequency band.

    High target-witness coherence channels receive larger channel weights.
    """

    def __init__(
        self,
        fs,
        fl,
        fh,
        fftlength=1.,
        overlap=None,
        reduction='mean',
        device='cpu',
        average='mean',
        coh_floor=1e-3,
        weight_by_target_coh=True,
        min_weight_sum=1e-8,
        use_log=True,

        # Channel-level weighting
        channel_coh_threshold=0.03,
        channel_weight_power=1.0,
        min_channel_weight_sum=1e-8,
        fallback_to_uniform_channels=True,

        # Internal computational chunking.
        # This does NOT reduce the number of witness channels used in the loss.
        channel_chunk_size=8,
    ):
        super().__init__()

        self.fs = fs
        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap else None
        self.reduction = reduction
        self.device = device
        self.average = average

        self.coh_floor = float(coh_floor)
        self.weight_by_target_coh = bool(weight_by_target_coh)
        self.min_weight_sum = float(min_weight_sum)
        self.use_log = bool(use_log)

        self.channel_coh_threshold = float(channel_coh_threshold)
        self.channel_weight_power = float(channel_weight_power)
        self.min_channel_weight_sum = float(min_channel_weight_sum)
        self.fallback_to_uniform_channels = bool(fallback_to_uniform_channels)

        self.channel_chunk_size = int(channel_chunk_size)
        if self.channel_chunk_size <= 0:
            raise ValueError("channel_chunk_size must be positive")

        freq = torch.linspace(
            0.,
            fs / 2.,
            self.nperseg // 2 + 1,
            device=device,
        )

        self.dfreq = freq[1] - freq[0]

        freq_mask = torch.zeros_like(
            freq,
            dtype=torch.uint8,
            device=device,
        )

        if not isinstance(fl, (list, tuple)):
            fl = [fl]
        if not isinstance(fh, (list, tuple)):
            fh = [fh]

        self.scale = 0.0

        for l, h in zip(fl, fh):
            band = ((freq >= l) & (freq <= h)).to(torch.uint8)
            freq_mask = torch.max(freq_mask, band)
            self.scale += (h - l)

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
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted noise, shape (B, T)

        target : torch.Tensor
            Target strain, shape (B, T)

        witness : torch.Tensor
            Witness channels, shape (B, C, T)
        """

        residual = target - pred

        B, C, T = witness.shape

        if C == 0:
            zero = torch.sum(pred * 0.0)
            return zero

        # Frequency mask: shape (1, 1, F)
        mask = self.freq_mask_float.view(1, 1, -1)

        # ------------------------------------------------------------
        # Shared spectra.
        # Srr needs gradient because residual depends on pred.
        # Stt is only a reference, so no_grad is safe.
        # ------------------------------------------------------------
        Srr = self._welch_1d(residual)  # (B, F)

        with torch.no_grad():
            Stt = self._welch_1d(target)  # (B, F)

        Srr = Srr.view(B, 1, -1)
        Stt = Stt.view(B, 1, -1)

        # Streaming accumulators.
        weighted_num = torch.zeros(B, device=self.device).type_as(pred)
        weighted_den = torch.zeros(B, device=self.device).type_as(pred)

        uniform_num = torch.zeros(B, device=self.device).type_as(pred)
        uniform_count = 0

        # ------------------------------------------------------------
        # Process all channels, but in chunks.
        # This lowers memory peak but does not remove any channel.
        # ------------------------------------------------------------
        for start in range(0, C, self.channel_chunk_size):
            end = min(start + self.channel_chunk_size, C)

            w_chunk = witness[:, start:end, :]  # (B, K, T)
            K = end - start

            # --------------------------------------------------------
            # Reference quantities do not need gradients.
            # --------------------------------------------------------
            with torch.no_grad():
                Sww = self._welch_channels(w_chunk)  # (B, K, F)

                Stw_r, Stw_i = self._cross_x_channels(
                    target,
                    w_chunk,
                )

                Stw_mag2 = Stw_r ** 2 + Stw_i ** 2

                coh_tgt = Stw_mag2 / (Stt * Sww + 1e-12)

                # Detach explicitly: target coherence is a reference.
                coh_tgt_ref = coh_tgt.detach()

            # --------------------------------------------------------
            # Residual-witness coherence.
            # This path must keep gradient through residual/pred.
            # --------------------------------------------------------
            Srw_r, Srw_i = self._cross_x_channels(
                residual,
                w_chunk,
            )

            Srw_mag2 = Srw_r ** 2 + Srw_i ** 2

            # Sww is a no_grad reference. Srr carries gradient.
            coh_res = Srw_mag2 / (Srr * Sww.detach() + 1e-12)

            denom = torch.clamp(
                coh_tgt_ref,
                min=self.coh_floor,
            )

            ratio = coh_res / denom

            if self.use_log:
                ratio = torch.log1p(ratio)

            # --------------------------------------------------------
            # Frequency-level weighting inside the target band.
            # Shape:
            #   ratio        : (B, K, F)
            #   freq_weights : (B, K, F)
            # --------------------------------------------------------
            if self.weight_by_target_coh:
                freq_weights = coh_tgt_ref * mask

                freq_weight_sum = torch.sum(
                    freq_weights,
                    dim=2,
                    keepdim=True,
                )

                uniform_freq_weights = mask.expand_as(freq_weights)

                fallback_freq = (
                    freq_weight_sum <= self.min_weight_sum
                ).type_as(freq_weights)

                freq_weights = (
                    freq_weights * (1.0 - fallback_freq)
                    + uniform_freq_weights * fallback_freq
                )
            else:
                freq_weights = mask.expand_as(ratio)

            freq_weight_sum = torch.sum(freq_weights, dim=2) + 1e-12

            # Per-channel loss for this chunk, shape (B, K)
            loss_chunk = torch.sum(
                ratio * freq_weights,
                dim=2,
            ) / freq_weight_sum

            # --------------------------------------------------------
            # Channel-level weighting.
            # Use target-witness coherence strength inside the band.
            # Shape: (B, K)
            # --------------------------------------------------------
            channel_strength = torch.sum(
                coh_tgt_ref * mask,
                dim=2,
            ) / (self.n_band_bins + 1e-12)

            channel_strength = channel_strength.detach()

            channel_weight = torch.clamp(
                channel_strength - self.channel_coh_threshold,
                min=0.0,
            )

            channel_weight = channel_weight ** self.channel_weight_power
            channel_weight = channel_weight.detach()

            # --------------------------------------------------------
            # Streaming accumulation.
            # Do not store all channel losses in a list.
            # --------------------------------------------------------
            weighted_num = weighted_num + torch.sum(
                loss_chunk * channel_weight,
                dim=1,
            )

            weighted_den = weighted_den + torch.sum(
                channel_weight,
                dim=1,
            )

            uniform_num = uniform_num + torch.sum(
                loss_chunk,
                dim=1,
            )

            uniform_count += K

        # ------------------------------------------------------------
        # Final channel aggregation.
        # ------------------------------------------------------------
        weighted_loss = weighted_num / (weighted_den + 1e-12)
        uniform_loss = uniform_num / (float(uniform_count) + 1e-12)

        if self.fallback_to_uniform_channels:
            fallback_channel = (
                weighted_den <= self.min_channel_weight_sum
            ).type_as(weighted_loss)

            loss = (
                weighted_loss * (1.0 - fallback_channel)
                + uniform_loss * fallback_channel
            )
        else:
            no_channel_weight = (
                weighted_den <= self.min_channel_weight_sum
            ).type_as(weighted_loss)

            loss = weighted_loss * (1.0 - no_channel_weight)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class TransferFunctionLoss(nn.Module):
    """
    Band-limited transfer-function loss.

    Goal
    ----
    Reduce the linear transfer-function-like coupling between
    residual strain and each witness channel.

    residual = target - pred

    For each witness channel w:

        H_res(f) = S_res,w(f) / S_w,w(f)
        H_tgt(f) = S_tgt,w(f) / S_w,w(f)

    In ratio mode, minimize:

        |H_res(f)|^2 / |H_tgt(f)|^2

    inside the target frequency band.

    This is useful when the target noise is approximately linearly
    coupled from witness channels.
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
        grad_boost=10.0,
        auto_grad_boost=True,
        max_grad_boost=1e3,
        syy_floor_factor=1e-6,
        tf_floor_factor=1e-6,
        center_weighting=True,
        sigma_fraction=4.0,
    ):
        super().__init__()

        self.fs = fs
        self.device = device
        self.average = average
        self.reduction = reduction
        self.nonlinearity = nonlinearity
        self.mode = mode

        self.grad_boost = float(grad_boost)
        self.auto_grad_boost = bool(auto_grad_boost)
        self.max_grad_boost = float(max_grad_boost)
        self.grad_target = 1.0
        self.ema_beta = 0.9

        self.syy_floor_factor = float(syy_floor_factor)
        self.tf_floor_factor = float(tf_floor_factor)
        self.center_weighting = bool(center_weighting)
        self.sigma_fraction = float(sigma_fraction)

        # EMA state for adaptive gradient scaling.
        self.register_buffer(
            "gb_ema",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "gb_inited",
            torch.tensor(0, dtype=torch.uint8, device=device),
        )

        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap else None

        freqs = torch.linspace(
            0.,
            fs / 2.,
            self.nperseg // 2 + 1,
            device=device,
        )

        self.dfreq = freqs[1] - freqs[0]

        freq_mask = torch.zeros_like(
            freqs,
            dtype=torch.uint8,
            device=device,
        )

        # Important fix:
        # Start from zeros, not ones.
        # If initialized with ones, max(ones, gaussian) is always ones,
        # so Gaussian center weighting has no effect.
        freq_weights = torch.zeros_like(freqs, device=device)

        if isinstance(fl, (float, int)):
            fl = [fl]
        if isinstance(fh, (float, int)):
            fh = [fh]

        self.scale = 0.0

        for l, h in zip(fl, fh):
            band = ((freqs >= l) & (freqs <= h)).to(torch.uint8)
            freq_mask = torch.max(freq_mask, band)

            self.scale += (h - l)

            if self.center_weighting:
                center = (l + h) / 2.0
                width = max(h - l, 1e-6)
                sigma = width / self.sigma_fraction

                gaussian = torch.exp(
                    -0.5 * ((freqs - center) / sigma) ** 2
                )

                freq_weights = torch.max(freq_weights, gaussian)
            else:
                freq_weights = torch.max(freq_weights, band.float())

        # If something went wrong, fallback to uniform band weighting.
        weight_in_mask = freq_weights * freq_mask.float()

        if torch.sum(weight_in_mask) <= 0:
            weight_in_mask = freq_mask.float()

        # Normalize such that sum(weight * df) = 1.
        denom = torch.sum(weight_in_mask * self.dfreq) + 1e-12
        weight_in_mask = weight_in_mask / denom

        self.register_buffer("freqs", freqs)
        self.register_buffer("freq_mask", freq_mask)
        self.register_buffer("freq_mask_float", freq_mask.float())
        self.register_buffer("weight_in_mask", weight_in_mask)

    def _welch(self, x):
        return _torch_welch(
            x,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def _cross_welch(self, x, y):
        return _torch_cross_welch(
            x,
            y,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            average=self.average,
            device=self.device,
        )

    def apply_nonlinearity(self, x):
        if self.nonlinearity == 'log1p':
            return torch.log1p(x)
        elif self.nonlinearity == 'sqrt':
            return torch.sqrt(x + 1e-12)
        elif self.nonlinearity == 'none':
            return x
        else:
            return x

    def _compute_grad_scale(self, x):
        """
        Adaptive gradient scaling.

        The forward value is preserved, but the gradient can be scaled.
        This helps when TF loss is numerically small.

        This should still be used carefully. Start with very small
        TF_WEIGHT, e.g. 0.001 to 0.01.
        """

        if not self.auto_grad_boost:
            return self.grad_boost

        with torch.no_grad():
            cur_med = x.detach().median()

            if self.gb_inited.item() == 0:
                self.gb_ema.copy_(
                    cur_med.clamp(min=1e-12).float()
                )
                self.gb_inited.fill_(1)

            self.gb_ema.mul_(self.ema_beta).add_(
                (1.0 - self.ema_beta) * cur_med.float()
            )

            scale = (
                self.grad_target * self.grad_boost
            ) / (float(self.gb_ema.item()) + 1e-12)

            scale = max(1.0, min(scale, self.max_grad_boost))

            return scale

    def forward(self, pred, target, witness):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted noise, shape (B, T)

        target : torch.Tensor
            Target strain, shape (B, T)

        witness : torch.Tensor
            Witness channels, shape (B, C, T)
        """

        residual = target - pred
        B, C, T = witness.shape

        weights = self.weight_in_mask.view(1, -1)

        channel_losses = []

        for i in range(C):
            w = witness[:, i, :]

            # Cross spectra
            Rxy_r, Rxy_i = self._cross_welch(residual, w)
            Txy_r, Txy_i = self._cross_welch(target, w)

            # Witness auto spectrum
            Syy = self._welch(w)

            # Adaptive floor for witness auto spectrum.
            syy_band = torch.sum(
                Syy * weights,
                dim=1,
                keepdim=True,
            )

            syy_floor = (
                self.syy_floor_factor * syy_band
                + 1e-12
            ).type_as(Syy)

            Syy_safe = torch.max(
                Syy,
                syy_floor.expand_as(Syy),
            )

            # |H_res|^2
            Rtf = (
                (Rxy_r / Syy_safe) ** 2
                + (Rxy_i / Syy_safe) ** 2
            )

            if self.mode == 'ratio':
                # |H_target|^2
                Ttf = (
                    (Txy_r / Syy_safe) ** 2
                    + (Txy_i / Syy_safe) ** 2
                )

                Ttf_ref = Ttf.detach()

                # Adaptive floor for target transfer function.
                ttf_band = torch.sum(
                    Ttf_ref * weights,
                    dim=1,
                    keepdim=True,
                )

                ttf_floor = (
                    self.tf_floor_factor * ttf_band
                    + 1e-12
                ).type_as(Ttf)

                Ttf_safe = torch.max(
                    Ttf_ref,
                    ttf_floor.expand_as(Ttf_ref),
                )

                x = Rtf / Ttf_safe

            elif self.mode == 'abs':
                x = Rtf

            else:
                raise ValueError(
                    "`mode` must be either 'ratio' or 'abs'"
                )

            # Band-limited weighted average.
            x = self.apply_nonlinearity(x)

            x_weighted = x * weights

            # Scale gradient only, not the forward value.
            scale = self._compute_grad_scale(x_weighted)
            x_weighted = _grad_scale(x_weighted, scale)

            loss_i = torch.sum(x_weighted, dim=1)

            channel_losses.append(loss_i)

        loss = torch.stack(channel_losses, dim=1).mean(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class CompositePSDLoss(nn.Module):
    def __init__(self, fs, fl, fh, fftlength=1.0, overlap=None, reduction='mean',
                 device='cpu', psd_weight=0.3, mse_weight=0.2, tf_weight=0.3,
                 coh_weight=0.2, average='mean', nonlinearity='log1p'):
        super().__init__()
        self.reduction = reduction
        self.psd_weight = psd_weight
        self.mse_weight = mse_weight
        self.coh_weight = coh_weight
        self.tf_weight = tf_weight
        self.device = device

        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.psd_loss = PSDLoss(fs, fl, fh, fftlength, overlap, reduction=reduction,
                                device=device, average=average)
        self.coh_loss = CoherenceLoss(fs, fl, fh, fftlength, overlap, reduction=reduction,
                                      device=device, average=average)
        self.tf_loss = TransferFunctionLoss(fs, fl, fh, fftlength, overlap, reduction=reduction,
                                            device=device, average=average, nonlinearity=nonlinearity)
        self.latest_loss_values = {
            'mse': 0.0, 'psd': 0.0, 'coh': 0.0, 'tf': 0.0, 'total': 0.0}

    def forward(self, pred, target, witness, return_dict=False):
        loss = 0.0
        comps = {}
        if self.mse_weight > 0:
            v = self.mse_loss(pred, target)
            loss += self.mse_weight * v
            comps['mse'] = v.item()
        if self.psd_weight > 0:
            v = self.psd_loss(pred, target)
            loss += self.psd_weight * v
            comps['psd'] = v.item()
        if self.coh_weight > 0:
            v = self.coh_loss(pred, target, witness)
            loss += self.coh_weight * v
            comps['coh'] = v.item()
        if self.tf_weight > 0:
            v = self.tf_loss(pred, target, witness)
            loss += self.tf_weight * v
            comps['tf'] = v.item()
        self.latest_loss_values.update(comps)
        self.latest_loss_values['total'] = loss.item() if isinstance(
            loss, torch.Tensor) else float(loss)
        return (loss, self.latest_loss_values) if return_dict else loss
