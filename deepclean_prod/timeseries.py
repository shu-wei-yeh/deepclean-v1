from fractions import Fraction
from scipy.signal import resample_poly

import h5py
import copy
from collections import OrderedDict

import numpy as np
import glob
import torch

from gwpy.timeseries import TimeSeriesDict

from .signal import bandpass


class TimeSeriesDataset:
    """ Torch dataset in timeseries format """

    def __init__(self):
        """ Initialized attributes """
        self.data = []
        self.channels = []
        self.t0 = 0.
        self.fs = 1.
        self.target_idx = None

    def fetch(self, channels, t0, duration, fs, nproc=1):
        """ Fetch data """
        # if channels is a file
        if isinstance(channels, str):
            channels = open(channels).read().splitlines()
        target_channel = channels[0]

        source_gwf = sorted(
            glob.glob('/home/shuwei.yeh/data_bank/o4c/*/*.gwf')
        )

        data = TimeSeriesDict.read(
            source=source_gwf,
            channels=channels,
            start=t0,
            end=t0 + duration,
            nproc=nproc
        )

        target_fs = float(fs)
        expected_nsamp = int(round(duration * target_fs))

        ## reset attributes ##
        self.data = []
        self.channels = []

        for chan, ts in data.items():
            current_fs = float(ts.sample_rate.value)

            if abs(current_fs - target_fs) < 1e-6:
                print(
                    f"[INFO] Skip resample for {chan}: already {current_fs} Hz")
                arr = ts.value
            else:
                print(
                    f"Resample {chan}: {current_fs} Hz to {target_fs} Hz using resample_poly")

                ratio = Fraction(int(round(target_fs)), int(
                    round(current_fs))).limit_denominator()
                up = ratio.numerator
                down = ratio.denominator

                arr = resample_poly(ts.value, up, down)

            # Make sure all channels have exactly the same length.
            # This avoids np.stack errors due to +/- 1 sample after resampling.
            if len(arr) > expected_nsamp:
                arr = arr[:expected_nsamp]
            elif len(arr) < expected_nsamp:
                pad = expected_nsamp - len(arr)
                arr = np.pad(arr, (0, pad), mode="edge")

            self.data.append(arr)
            self.channels.append(chan)

        self.data = np.stack(self.data)
        self.channels = np.stack(self.channels)
        self.t0 = t0
        self.fs = fs
        self.target_idx = np.where(self.channels == target_channel)[0][0]

    def read(self, fname, channels, group=None):
        """ Read data from HDF5 format """
        # if channels is a file
        if isinstance(channels, str):
            channels = open(channels).read().splitlines()
        target_channel = channels[0]

        # read data from HDF5 file
        self.data = []
        self.channels = []
        with h5py.File(fname, 'r') as f:
            if group is not None:
                fobj = f[group]
            else:
                fobj = f

            for chan, data in fobj.items():
                if chan not in channels:
                    continue
                self.channels.append(chan)
                self.data.append(data[:])
                self.t0 = data.attrs['t0']
                self.fs = data.attrs['sample_rate']
        self.data = np.stack(self.data)
        self.channels = np.stack(self.channels)

        # sorted by channel name
        sorted_indices = np.argsort(self.channels)
        self.channels = self.channels[sorted_indices]
        self.data = self.data[sorted_indices]
        self.target_idx = np.where(self.channels == target_channel)[0][0]

    def write(self, fname, group=None, write_mode='w'):
        """ Write to HDF5 format. Can be read directly by gwpy.timeseries.TimeSeriesDict """
        with h5py.File(fname, write_mode) as f:
            # write to group if group is given
            if group is not None:
                fobj = f.create_group(group)
            else:
                fobj = f
            for chan, ts in zip(self.channels, self.data):
                dset = fobj.create_dataset(chan, data=ts, compression='gzip')
                dset.attrs['sample_rate'] = self.fs
                dset.attrs['t0'] = self.t0
                dset.attrs['channel'] = str(chan)
                dset.attrs['name'] = str(chan)

    def bandpass(self, fl, fh, order=8, channels=None):
        """ Bandpass filter data """
        if isinstance(fl, (list, tuple)):
            fl = fl[0]
        if isinstance(fh, (list, tuple)):
            fh = fh[-1]

        # create a copy of the class
        new = self.copy()

        # bandpassing
        if isinstance(channels, str):
            if channels == 'all':
                new.data = bandpass(new.data, self.fs, fl, fh, order)
            elif channels == 'target':
                new.data[new.target_idx] = bandpass(
                    new.data[new.target_idx], self.fs, fl, fh, order)
            elif channels == 'aux':
                for i, d in enumerate(new.data):
                    if i == new.target_idx:
                        continue
                    new.data[i] = bandpass(d, self.fs, fl, fh, order)
        elif isinstance(channels, list):
            for i, (chan, d) in enumerate(zip(new.channels, new.data)):
                if chan not in channels:
                    continue
                new.data[i] = bandpass(d, self.fs, fl, fh, order)

        return new

    def normalize(self, mean=None, std=None):
        """ Normalize data by mean and std """
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std

        new = self.copy()
        new.data = (new.data - mean) / std
        return new

    def copy(self):
        """ Return a copy of class """
        return copy.deepcopy(self)

    def get(self, channels):
        """ Return data from given channels """
        data = []
        for chan, d in zip(self.channels, self.data):
            if chan not in channels:
                continue
            data.append(d)
        data = np.stack(data)
        return data

    def get_target(self):
        """ Get target channel """
        return self.data[self.target_idx]

    @property
    def mean(self):
        """ Return mean of each channel """
        return self.data.mean(axis=-1, keepdims=True)

    @property
    def std(self):
        """ Return std of each channel """
        return self.data.std(axis=-1, keepdims=True)

    @property
    def n_channels(self):
        """ Return number of channels 2"""
        return len(self.channels)


class TimeSeriesSegmentDataset(TimeSeriesDataset):
    """ Torch timeseries dataset with segment """

    def __init__(self, kernel, stride, pad_mode='median'):

        super().__init__()

        self.kernel = kernel
        self.stride = stride
        self.pad_mode = pad_mode

    def __len__(self):
        """ Return the number of stride """
        nsamp = self.data.shape[-1]
        kernel = int(self.kernel * self.fs)
        stride = int(self.stride * self.fs)
        n_stride = int(np.ceil((nsamp - kernel) / stride) + 1)
        return max(0, n_stride)

    def __getitem__(self, idx):
        """ Get sample Tensor for a given index """
        if idx < 0:
            idx += self.__len__()
        if idx >= self.__len__():
            raise IndexError(
                f'index {idx} is out of bound with size {self.__len__()}.')

        kernel = int(self.kernel * self.fs)
        stride = int(self.stride * self.fs)
        idx_start = idx * stride
        idx_stop = idx_start + kernel
        data = self.data[:, idx_start: idx_stop].copy()

        # padding if needed
        nsamp = data.shape[-1]
        if nsamp < kernel:
            pad = kernel - nsamp
            data = np.pad(data, ((0, 0), (0, pad)), mode=self.pad_mode)

        # separate target and aux, then re-stack: [aux, ..., target]
        target = data[self.target_idx]
        aux = np.delete(data, self.target_idx, axis=0)

        # Concatenate so that target is the last channel
        data_tensor = np.concatenate([aux, target[None, :]], axis=0)

        # dummy target for compatibility
        return torch.Tensor(data_tensor), torch.tensor(0)
