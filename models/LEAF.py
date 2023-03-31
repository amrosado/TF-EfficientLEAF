import tensorflow as tf
import numpy as np
from typing import Optional

def return_peaks_mel(n_filters, min_freq, max_freq):
    min_mel = 1127 * tf.math.log1p(min_freq / 700.0)
    max_mel = 1127 * tf.math.log1p(max_freq / 700.0)
    peaks_mel = tf.linspace(min_mel, max_mel, n_filters + 2)

    return peaks_mel

class MelFilterCenterFreqs(tf.keras.initializers.Initializer):
    def __init__(self, n_filters: int, min_freq: float, max_freq: float, sample_rate: int) -> tf.Tensor:
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate

    def __call__(self, shape, dtype=None, **kwargs):
        peaks_hz = return_peaks_mel(self.n_filters, self.min_freq, self.max_freq)
        center_freqs = peaks_hz[1:-1] * (2 * np.pi / self.sample_rate)

        return center_freqs

class MelFilterBandwidths(tf.keras.initializers.Initializer):
    def __init__(self, n_filters: int, min_freq: float, max_freq: float, sample_rate: int) -> tf.Tensor:
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate

    def __call__(self, shape, dtype=None, **kwargs):
        peaks_hz = return_peaks_mel(self.n_filters, self.min_freq, self.max_freq)
        bandwidths = peaks_hz[2:] - peaks_hz[:-2]
        sigmas = (self.sample_rate / 2.) / bandwidths

        return sigmas

def mel_filter_params(n_filters: int, min_freq: float, max_freq: float, sample_rate: int) -> (tf.Tensor, tf.Tensor):
    min_mel = 1127 * tf.math.log1p(min_freq / 700.0)
    max_mel = 1127 * tf.math.log1p(min_freq / 700.0)
    peaks_mel = tf.linspace(min_mel, max_mel, n_filters + 2)
    peaks_hz = 700 * (tf.math.expm1(peaks_mel / 1127))
    center_freqs = peaks_hz[1:-1] * (2 * np.pi / sample_rate)
    bandwidths = peaks_hz[2:] - peaks_hz[:2]
    sigmas = (sample_rate / 2.) / bandwidths
    return center_freqs, sigmas

def gabor_filters(size: int, center_freqs: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
    t = tf.range(-(size // 2), (size + 1) // 2, dtype=tf.dtypes.float32)
    denominator = tf.dtypes.complex(sigmas * 1. / (np.sqrt(2 * np.pi)), tf.constant(0., dtype=tf.dtypes.float32))
    gaussian = tf.dtypes.complex(tf.math.exp(tf.tensordot(-t**2, 1. / (2. * sigmas**2), axes=0)), tf.constant(0., dtype=tf.dtypes.float32))
    outer_product = tf.dtypes.complex(tf.tensordot(t, center_freqs, axes=0), tf.constant(0., dtype=tf.dtypes.float32))
    sinusoid = tf.math.exp(outer_product * tf.constant(1j, dtype=tf.dtypes.complex64))
    return denominator[tf.newaxis, :] * sinusoid * gaussian

def gauss_windows(size: int, sigmas: tf.Tensor) -> tf.Tensor:
    t = tf.range(0, size, dtype=tf.dtypes.float32)
    numerator = (2 / (size - 1)) * t - 1
    sigmas = tf.expand_dims(sigmas, axis=1)
    return tf.transpose(tf.math.exp(-0.5 * (numerator / sigmas)**2))

class PCEN(tf.keras.Model):
    def __init__(self, num_bands: int, s: float=0.025, alpha: float=1.,
                 delta: float=1., r: float=1., eps: float=1e-6, learn_logs: bool=True,
                 clamp: Optional[float]=None):
        super(PCEN, self).__init__()
        if learn_logs:
            s = tf.math.log(s)
            alpha = tf.math.log(alpha)
            delta = tf.math.log(delta)
            r = tf.math.log(r)
        else:
            r = 1. / r

        self.learn_logs = learn_logs
        self.s = self.add_weight('pcen_s', shape=num_bands, initializer=tf.keras.initializers.Constant(s), trainable=True)
        self.alpha = self.add_weight('pcen_alpha', shape=num_bands, initializer=tf.keras.initializers.Constant(alpha), trainable=True)
        self.delta = self.add_weight('pcen_delta', shape=num_bands, initializer=tf.keras.initializers.Constant(delta), trainable=True)
        self.r = self.add_weight('pcen_r', shape=num_bands, initializer=tf.keras.initializers.Constant(r), trainable=True)
        self.eps = tf.Tensor(eps)
        self.clamp = clamp

    def call(self, input: tf.Tensor):
        if self.clamp is not None:
            input = tf.clip_by_value(input, clip_value_min=self.clamp)

        if self.learn_logs:
            s = tf.math.exp(self.s)
            alpha = tf.math.exp(self.alpha)
            delta = tf.math.exp(self.delta)
            r = tf.math.exp(self.r)
        else:
            s = self.s
            alpha = self.alpha
            detla = self.delta
            r = 1. / tf.clip_by_value(self.r, clip_value_min=1)

        alpha = alpha[:, tf.newaxis]
        delta = delta[:, tf.newaxis]
        r = r[:, tf.newaxis]

        smoother = [input[..., 0]]

        for frame in range(1, input.shape[-1]):
            smoother.append((1 - s) * smoother[-1] + s * input[..., frame])
        smoother = tf.stack(smoother, -1)

        smoother = tf.math.exp(-alpha * (tf.math.log(self.eps) +
                                         tf.math.log1p(smoother / self.eps)))

        return (input * smoother + delta)**r - delta**r

class LEAF(tf.keras.Model):
    def __init__(self):
        super(LEAF, self).__init__()