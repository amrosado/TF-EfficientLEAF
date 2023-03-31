from typing import Optional, Union
import tensorflow as tf
import numpy as np

from models.LEAF import mel_filter_params, gabor_filters, gauss_windows
from models.LEAF import PCEN
from models.LEAF import MelFilterBandwidths, MelFilterCenterFreqs

import matplotlib

def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)

class LogTBN(tf.keras.Model):
    def __init__(self, num_bands: int, affine: bool=True, a: float=0, trainable: bool=False,
                 per_band: bool=False, median_filter: bool=False, append_filtered: bool=False):
        super(LogTBN, self).__init__()
        self.log1p = Log1p(a=a, trainable=trainable, per_band=per_band, num_bands=num_bands)
        self.TBN = TemporalBatchNorm(num_bands=num_bands, affine=affine, per_channel=append_filtered, num_channels=2 if append_filtered else 1)
        self.median_filter = median_filter
        self.append_filtered = append_filtered

    def call(self, x: tf.Tensor):
        x = self.log1p(x)
        if self.median_filter:
            if self.append_filtered and len(x.shape) == 3:
                x = tf.expand_dims(x, axis=3)
            m = get_median(x)
            if self.append_filtered:
                x = tf.concat((x, x - m), axis=3)
            else:
                x = x - m

        x = self.TBN(x)

        return x

class Log1p(tf.keras.Model):
    def __init__(self, a=0, trainable=False, per_band=False, num_bands=None):
        super(Log1p, self).__init__()

        if trainable:
            dtype = tf.dtypes.float32
            if not per_band:
                a = tf.Tensor(a, dtype=dtype)
            else:
                a = self.add_weight('Log1p_a', shape=num_bands, initializer=tf.keras.initializers.Constant(a), trainable=True)
        self.a = a
        self.trainable = trainable
        self.per_band = per_band

    def call(self, x):
        if self.trainable or self.a != 0:
            a = self.a[tf.newaxis, :] if self.per_band else self.a
            x = 10 ** a * x
        return tf.math.log1p(x)

    def extra_repr(self):
        return 'trainable={}, per_band={}'.format(repr(self.trainable), repr(self.per_band))

class TemporalBatchNorm(tf.keras.Model):
    def __init__(self, num_bands: int, affine:bool = True, per_channel: bool=False,
                 num_channels: Optional[int] = None):
        super(TemporalBatchNorm, self).__init__()
        num_features = num_bands * num_channels if per_channel else num_bands
        self.bn = tf.keras.layers.BatchNormalization(axis=1)
        self.per_channel = per_channel

    def call(self, x):
        shape = x.shape
        if self.per_channel:
            x = tf.reshape(x, (shape[0], shape[1], -1))
        else:
            x = tf.reshape(x, ((-1,) + x.shape[-2:]))

        # x = tf.transpose(x, [0, 2, 1])
        x = self.bn(x)
        # x = tf.transpose(x, [0, 2, 1])

        return tf.reshape(x, shape)

class GroupedGaborFilterbank(tf.keras.Model):
    """
    Tensorflow Keras module that functions as a gabor filterbank. Heavily re-created from original pytorch EfficientLEAF
    implementation. Initializes n_filters center frequencies
    and bandwidths that are based on a mel filterbank. The parameters are used to calculate Gabor filters
    for a 1D convolution over the input signal. The squared modulus is taken from the results.
    To reduce the temporal resolution a gaussian lowpass filter is calculated from pooling_widths,
    which are used to perform a pooling operation.
    The center frequencies, bandwidths and pooling_widths are learnable parameters.
    The module splits the different filters into num_groups and calculates for each group a separate kernel size
    and stride, so at the end all groups can be merged to a single output. conv_win_factor and stride_factor
    are parameters that can be used to influence the kernel size and stride.
    :param n_filters: number of filters
    :param num_groups: number of groups
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param pool_size: size of the kernels/filters for pooling convolution
    :param pool_stride: stride of the pooling convolution
    :param pool_init: initial value for the gaussian lowpass function
    :param conv_win_factor: factor is multiplied with the kernel/filter size
    :param stride_factor: factor is multiplied with the kernel/filter stride
    """
    def __init__(self,
                 n_filters: int,
                 num_groups: float,
                 min_freq: float,
                 max_freq: float,
                 sample_rate: int,
                 pool_size: int,
                 pool_stride: int,
                 pool_init: float=0.4,
                 conv_win_factor: float=3.,
                 stride_factor: float=1.,
                 dynamic=True
                 ):
        super(GroupedGaborFilterbank, self).__init__(dynamic=dynamic)
        self.num_groups = num_groups
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.conv_win_factor = conv_win_factor
        self.stride_factor = stride_factor
        self.possible_strides = [i for i in range(1, pool_stride+1) if pool_stride % i == 0]

        self.center_freqs = self.add_weight('grouped_gabor_center_freqs', shape=n_filters, initializer=MelFilterCenterFreqs(n_filters, min_freq, max_freq, sample_rate), trainable=True, dtype=tf.dtypes.float32)
        self.bandwidths = self.add_weight('grouped_gabor_center_freqs', shape=n_filters, initializer=MelFilterBandwidths(n_filters, min_freq, max_freq, sample_rate), trainable=True, dtype=tf.dtypes.float32)
        self.pooling_widths = self.add_weight('grouped_gabor_pooling', shape=n_filters, initializer=tf.keras.initializers.Constant(pool_init), trainable=True, dtype=tf.dtypes.float32)

        self.mu_lower = tf.constant(0.)
        self.mu_upper = tf.constant(np.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        self.sigma_lower = tf.constant(2 * z, dtype=tf.float32)
        self.sigma_upper = tf.constant(pool_size * z, dtype=tf.float32)

    def get_stride(self, cent_freq):
        '''
        Calculates the dynamic convolution and pooling stride, based on the max center frequency of the
        group. This ensures that the outputs for each group have the same dimensions.
        :param cent_freq: max center frequency
        '''
        stride = tf.maximum(1., np.pi / cent_freq * self.stride_factor)
        tf_poss = tf.constant(self.possible_strides, dtype=tf.float32)
        search_sorted = tf.searchsorted(tf_poss, [stride], side='right')[0]
        stride = tf.gather(tf_poss, search_sorted - 1).numpy()
        return stride, self.pool_stride // stride

    def clamp_parameters(self):
        '''
        Clamps the center frequencies, bandwidth and pooling widths.
        '''
        self.center_freqs.assign(tf.clip_by_value(self.center_freqs, clip_value_min=self.mu_lower, clip_value_max=self.mu_upper))
        self.bandwidths.assign(tf.clip_by_value(self.bandwidths, clip_value_min=self.sigma_lower, clip_value_max=self.sigma_upper))
        self.pooling_widths.assign(tf.clip_by_value(self.pooling_widths, clip_value_min=2./self.pool_size, clip_value_max=0.5))

    def call(self, x):
        # constraint center frequencies and pooling widths
        self.clamp_parameters()
        bandwidths = self.bandwidths
        center_freqs = self.center_freqs

        # iterate over groups
        splits = np.arange(self.num_groups + 1) * self.n_filters // self.num_groups
        outputs = []

        for i, (a, b) in enumerate(zip(splits[:-1], splits[1:])):
            num_group_filters = b - a
            # calculate strides
            conv_stride, pool_stride = self.get_stride(tf.math.reduce_max(center_freqs[a:b]))

            # complex convolution
            ## compute filters
            kernel_size = int(np.max(bandwidths[a:b]) * self.conv_win_factor)
            kernel_size += 1 - kernel_size % 2
            kernel = gabor_filters(kernel_size, center_freqs[a:b], bandwidths[a:b])
            # kernel = tf.transpose(kernel)
            kernel = tf.expand_dims(tf.concat([tf.math.real(kernel), tf.math.imag(kernel)], axis=1), axis=1)
            # kernel = tf.expand_dims(tf.concat([tf.math.real(kernel), tf.math.imag(kernel)], axis=0), axis=1)
            # kernel_trans = tf.transpose(kernel, [2, 1, 0])

            # compute squared modulus
            output = tf.pad(x, [[0,0], [kernel_size // 2, kernel_size // 2], [0,0]])
            # output_nwc = tf.transpose(output, [0, 2, 1])
            output = tf.nn.conv1d(output, kernel, stride=float(conv_stride), padding='VALID')
            # output_ncw = tf.transpose(output, [0, 2, 1])

            output = tf.math.square(output)
            output = output[:, :, :num_group_filters] + output[:, :, num_group_filters:]

            window_size = int(self.pool_size / conv_stride + .5)
            window_size += 1 - window_size % 2

            sigma = self.pooling_widths[a:b]/conv_stride * self.pool_size/window_size
            windows = tf.expand_dims(gauss_windows(window_size, sigma), axis=1)
            windows_trans = tf.transpose(windows, [2, 1, 0])

            # output_nwc = tf.transpose(output_ncw, [0, 2, 1])

            group_num = int(output.shape[2] // num_group_filters)

            group_output = []

            for i in range(group_num):
                group = output[:, :, i*num_group_filters:(i+1)*num_group_filters]
                group = tf.pad(group, [[0, 0], [window_size//2, window_size//2], [0, 0]])
                group = tf.nn.conv1d(group, windows, stride=pool_stride, padding='VALID')
                group_output.append(group)

            output = tf.concat(group_output, axis=2)

            # output_ncw = tf.transpose(output_nwc, [0, 2, 1])

            outputs.append(output)

        output = tf.concat(outputs, axis=2)

        return output

class EfficientLeaf(tf.keras.Model):
    def __init__(self,
                 n_filters: int=40,
                 num_groups: int=4,
                 min_freq: float=60.0,
                 max_freq: float=7800.0,
                 sample_rate: int=16000,
                 window_len: float=25.,
                 window_stride: float=10.,
                 conv_win_factor: float=4.77,
                 stride_factor: float=1.,
                 compression: Union[str, tf.keras.Model]='logtbn'
                 ):
        super(EfficientLeaf, self).__init__()

        window_size = int(sample_rate * window_len / 1000)
        window_size += (window_size % 2)
        window_stride = int(sample_rate * window_stride / 1000)

        self.filterbank = GroupedGaborFilterbank(
            n_filters, num_groups, min_freq,
            max_freq, sample_rate, pool_size=window_size,
            pool_stride=window_stride, conv_win_factor=conv_win_factor,
            stride_factor=stride_factor,
            dynamic=True
        )

        if compression == 'pcen':
            self.compression = PCEN(n_filters, s=0.04, alpha=0.96, delta=2, r=0.5, eps=1e-12, learn_logs=False, clamp=1e-5)

        elif compression == 'logtbn':
            self.compression = LogTBN(n_filters, a=5., trainable=True, per_band=True, median_filter=True, append_filtered=True)

        elif isinstance(compression, tf.keras.Model):
            self.compression = compression
        else:
            raise ValueError("unsupported value for compression argument")

    def call(self, x: tf.Tensor):
        while len(x.shape) < 3:
            x = tf.expand_dims(x, axis=2)
        x = self.filterbank(x)
        x = self.compression(x)
        return x