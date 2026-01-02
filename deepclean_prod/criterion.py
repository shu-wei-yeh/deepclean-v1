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
    def __init__(self, fs, fl, fh, fftlength=1., overlap=None,
                 reduction='mean', device='cpu', average='mean'):
        super().__init__()
        self.fs = fs
        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap else None
        self.reduction = reduction
        self.device = device
        self.average = average

        freq = torch.linspace(0., fs/2., self.nperseg//2 + 1, device=device)
        self.dfreq = freq[1] - freq[0]
        self.freq_mask = torch.zeros_like(
            freq, dtype=torch.uint8, device=device)
        self.scale = 0.0

        if not isinstance(fl, (list, tuple)):
            fl = [fl]
        if not isinstance(fh, (list, tuple)):
            fh = [fh]
        for l, h in zip(fl, fh):
            band = ((freq >= l) & (freq <= h)).to(torch.uint8)
            self.freq_mask = torch.max(self.freq_mask, band)
            self.scale += (h - l)

    def forward(self, pred, target, witness):
        residual = target - pred
        B, C, T = witness.shape
        ratio_list = []
        for i in range(C):
            w = witness[:, i, :]
            coh_res = compute_coherence(residual, w, self.fs, self.nperseg,
                                        self.noverlap, self.device, self.average)
            coh_tgt = compute_coherence(target, w, self.fs, self.nperseg,
                                        self.noverlap, self.device, self.average)
            # zero out-of-band
            mask_eq0 = (self.freq_mask == 0)
            coh_res[:, mask_eq0] = 0.0
            coh_tgt[:, mask_eq0] = 0.0

            ratio = coh_res / (coh_tgt + 1e-12)
            ratio_list.append(torch.mean(ratio, dim=1))  # (B,)

        loss = torch.stack(ratio_list, dim=1).mean(dim=1)  # (B,)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class TransferFunctionLoss(nn.Module):
    """
    TF loss with internal gradient scaling (median+EMA).
    mode: 'ratio' (default) uses |TF(residual)|^2 / |TF(target)|^2, or 'abs' uses |TF(residual)|^2.
    """

    def __init__(self, fs, fl, fh, fftlength=1.0, overlap=None,
                 reduction='mean', device='cpu', average='mean',
                 nonlinearity='log1p', mode='ratio',
                 grad_boost=100.0, auto_grad_boost=True, max_grad_boost=1e6):
        super().__init__()
        self.fs = fs
        self.device = device
        self.average = average
        self.reduction = reduction
        self.nonlinearity = nonlinearity
        self.mode = mode

        # Gradient scale controls
        self.grad_boost = float(grad_boost)      # base multiplier
        self.auto_grad_boost = bool(auto_grad_boost)
        self.max_grad_boost = float(max_grad_boost)
        self.grad_target = 1.0                   # push batch median*scale ~ 1
        self.ema_beta = 0.9                      # EMA smoothing

        # EMA state (buffers -> saved in ckpt, no grad)
        self.register_buffer("gb_ema", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("gb_inited", torch.tensor(0, dtype=torch.uint8))

        self.nperseg = int(fftlength * fs)
        self.noverlap = int(overlap * fs) if overlap else None

        freqs = torch.linspace(0., fs/2., self.nperseg // 2 + 1, device=device)
        self.freqs = freqs
        self.dfreq = freqs[1] - freqs[0]

        self.freq_mask = torch.zeros_like(
            freqs, dtype=torch.uint8, device=device)
        self.freq_weights = torch.ones_like(freqs, device=device)

        if isinstance(fl, (float, int)):
            fl = [fl]
        if isinstance(fh, (float, int)):
            fh = [fh]

        self.scale = 0.0
        for l, h in zip(fl, fh):
            band = ((freqs >= l) & (freqs <= h)).to(torch.uint8)
            self.freq_mask = torch.max(self.freq_mask, band)
            self.scale += (h - l)
            center = (l + h) / 2.0
            width = (h - l)
            self.freq_weights = torch.max(
                self.freq_weights,
                torch.exp(-0.5 * ((freqs - center) / (width / 4.0))**2)
            )

        # normalize weights inside band
        weight_in_mask = self.freq_weights * self.freq_mask.float()
        denom = torch.sum(weight_in_mask * self.dfreq) + 1e-12
        self.weight_in_mask = weight_in_mask / denom  # (F)

    def _welch(self, x):
        return _torch_welch(x, fs=self.fs, nperseg=self.nperseg,
                            noverlap=self.noverlap, average=self.average,
                            device=self.device)

    def _cross_welch(self, x, y):
        return _torch_cross_welch(x, y, fs=self.fs, nperseg=self.nperseg,
                                  noverlap=self.noverlap, average=self.average,
                                  device=self.device)

    def apply_nonlinearity(self, x):
        if self.nonlinearity == 'log1p':
            return torch.log1p(x)
        elif self.nonlinearity == 'sqrt':
            return torch.sqrt(x + 1e-12)
        else:
            return x

    def _compute_grad_scale(self, tf_masked):
        """
        Use robust batch statistic (median) + EMA to decide gradient scale.
        """
        if not self.auto_grad_boost:
            return self.grad_boost
        with torch.no_grad():
            cur_med = tf_masked.detach().median()  # scalar
            if self.gb_inited.item() == 0:
                self.gb_ema.copy_(cur_med.clamp(min=1e-12).float())
                self.gb_inited.fill_(1)
            # EMA smoothing
            self.gb_ema.mul_(self.ema_beta).add_(
                (1.0 - self.ema_beta) * cur_med)
            # target: med * scale ~ grad_target
            s = (self.grad_target * self.grad_boost) / \
                (float(self.gb_ema.item()) + 1e-12)
            s = max(1.0, min(s, self.max_grad_boost))
            return s

    def forward(self, pred, target, witness):
        residual = target - pred
        B, C, T = witness.shape
        vals = []
        for i in range(C):
            w = witness[:, i, :]

            # Cross & auto spectra
            Rxy_r, Rxy_i = self._cross_welch(residual, w)
            Txy_r, Txy_i = self._cross_welch(target,   w)
            Syy = self._welch(w)

            # adaptive floor on Syy inside band
            syy_band = torch.sum(Syy * self.weight_in_mask,
                                 dim=1, keepdim=True)  # (B,1)
            eps = (1e-6 * syy_band + 1e-12).type_as(Syy).expand_as(Syy)
            Syy = torch.max(Syy, eps)  # elementwise clamp (old torch-friendly)

            # |TF|^2
            Rtf = (Rxy_r / Syy)**2 + (Rxy_i / Syy)**2
            if self.mode == 'ratio':
                Ttf = (Txy_r / Syy)**2 + (Txy_i / Syy)**2
                x = Rtf / (Ttf + 1e-12)
            else:
                x = Rtf

            # weighting (band only) + optional nonlinearity
            x = self.apply_nonlinearity(x) * self.weight_in_mask  # (B, F)

            # scale gradient only
            scale = self._compute_grad_scale(x)  # scalar
            x = _grad_scale(x, scale)

            vals.append(torch.sum(x, dim=1))  # (B,)

        loss = torch.stack(vals, dim=1).mean(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


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
