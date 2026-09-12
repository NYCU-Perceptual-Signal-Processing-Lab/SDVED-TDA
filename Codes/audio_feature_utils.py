"""Dependency-light signal processing helpers for the 44-D feature extractor."""

import math
import wave

import numpy as np


LOW_FREQ_CUTOFF = 2000
N_MFCC = 13
SR = 48000
N_FFT = 2048
HOP_LENGTH = 512
MEL_BANDS = 128
EPS = np.finfo(np.float64).eps


def read_wav_mono(file_path):
    """Read a WAV file as mono float64 samples in the range [-1, 1]."""
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif sample_width == 3:
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            sign = (packed[:, 2] & 0x80) != 0
            extended = np.empty((packed.shape[0], 4), dtype=np.uint8)
            extended[:, :3] = packed
            extended[:, 3] = np.where(sign, 0xFF, 0x00)
            audio = extended.view("<i4").reshape(-1).astype(np.float64) / 8388608.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        audio = np.nan_to_num(audio, copy=False)
        return audio.astype(np.float64, copy=False), sample_rate
    except wave.Error:
        return read_audio_with_soundfile(file_path)


def read_audio_with_soundfile(file_path):
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot read {file_path}. It is not a standard RIFF WAV file, "
            "and soundfile is not installed for fallback decoding."
        ) from exc

    audio, sample_rate = sf.read(str(file_path), always_2d=True, dtype="float64")
    audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, copy=False)
    return audio.astype(np.float64, copy=False), int(sample_rate)


def sinc_resample(y, orig_sr, target_sr, radius=32, chunk_size=32768):
    if orig_sr == target_sr or y.size == 0:
        return y.astype(np.float64, copy=False)

    ratio = target_sr / float(orig_sr)
    out_len = max(1, int(round(y.size * ratio)))
    cutoff = min(1.0, ratio)
    padded = np.pad(y.astype(np.float64, copy=False), radius + 2, mode="edge")
    output = np.empty(out_len, dtype=np.float64)
    kernel_offsets = np.arange(-radius + 1, radius + 1, dtype=np.float64)

    for start in range(0, out_len, chunk_size):
        end = min(start + chunk_size, out_len)
        out_index = np.arange(start, end, dtype=np.float64)
        input_pos = out_index / ratio
        left = np.floor(input_pos).astype(np.int64)
        sample_index = left[:, None] + kernel_offsets.astype(np.int64)[None, :]
        distance = input_pos[:, None] - sample_index

        window = 0.5 + 0.5 * np.cos(np.pi * distance / radius)
        window[np.abs(distance) > radius] = 0.0
        kernel = cutoff * np.sinc(cutoff * distance) * window
        kernel_sum = kernel.sum(axis=1, keepdims=True)
        kernel = np.divide(
            kernel,
            kernel_sum,
            out=np.zeros_like(kernel),
            where=np.abs(kernel_sum) > EPS,
        )
        output[start:end] = np.sum(padded[sample_index + radius] * kernel, axis=1)

    return output


def resample_audio(y, orig_sr, target_sr):
    try:
        import soxr
    except ModuleNotFoundError:
        return sinc_resample(y, orig_sr, target_sr)
    return soxr.resample(y, orig_sr, target_sr, quality="HQ").astype(
        np.float64, copy=False
    )


def load_audio(file_path, target_sr=SR):
    y, sr = read_wav_mono(file_path)
    if sr != target_sr:
        y = resample_audio(y, sr, target_sr)
        sr = target_sr
    return y, sr


def periodic_hann(length):
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / float(length))


def frame_signal(y, frame_length, hop_length, center=True):
    if center:
        y = np.pad(y, frame_length // 2, mode="constant")
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size), mode="constant")

    frame_count = 1 + (y.size - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def stft_matrix(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True):
    frames = frame_signal(y, n_fft, hop_length, center=center)
    windowed = frames * periodic_hann(n_fft)[None, :]
    return np.fft.rfft(windowed, n=n_fft, axis=1).T


def magnitude_spectrogram(
    y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True, power=1.0
):
    spectrum = stft_matrix(y, n_fft=n_fft, hop_length=hop_length, center=center)
    magnitude = np.abs(spectrum)
    return magnitude if power == 1.0 else magnitude**power


def spectral_centroid_mean(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH):
    magnitude = magnitude_spectrogram(
        y, n_fft=n_fft, hop_length=hop_length, power=1.0
    )
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    denominator = magnitude.sum(axis=0)
    centroid = np.divide(
        freqs[:, None].T.dot(magnitude).reshape(-1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > EPS,
    )
    return float(np.mean(centroid)) if centroid.size else 0.0


def hz_to_mel(frequencies):
    frequencies = np.asanyarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mels[log_region] = min_log_mel + np.log(
        frequencies[log_region] / min_log_hz
    ) / logstep
    return mels


def mel_to_hz(mels):
    mels = np.asanyarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * np.exp(
        logstep * (mels[log_region] - min_log_mel)
    )
    return freqs


def mel_filterbank(sr, n_fft=N_FFT, n_mels=MEL_BANDS, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    weights = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lower, center, upper = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        left = (
            (fft_freqs - lower) / (center - lower)
            if center > lower
            else np.zeros_like(fft_freqs)
        )
        right = (
            (upper - fft_freqs) / (upper - center)
            if upper > center
            else np.zeros_like(fft_freqs)
        )
        weights[i] = np.maximum(0.0, np.minimum(left, right))

    enorm = 2.0 / np.maximum(
        hz_points[2 : n_mels + 2] - hz_points[:n_mels], EPS
    )
    weights *= enorm[:, None]
    return weights


def power_to_db(power, ref=1.0, amin=1e-10, top_db=80.0):
    power = np.asarray(power, dtype=np.float64)
    log_spec = 10.0 * np.log10(np.maximum(amin, power))
    if ref == "max":
        ref_value = np.max(power)
    elif callable(ref):
        ref_value = ref(power)
    else:
        ref_value = ref
    log_spec -= 10.0 * np.log10(max(amin, float(ref_value)))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def mel_spectrogram(
    y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=MEL_BANDS, fmax=None
):
    power = magnitude_spectrogram(
        y, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    mel_basis = mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    return np.maximum(mel_basis.dot(power), 0.0)


def dct_type_2_ortho(data, n_coeffs):
    n = data.shape[0]
    basis = np.cos(
        np.pi
        / n
        * (np.arange(n, dtype=np.float64) + 0.5)[None, :]
        * np.arange(n_coeffs, dtype=np.float64)[:, None]
    )
    basis[0] *= math.sqrt(1.0 / (4.0 * n))
    if n_coeffs > 1:
        basis[1:] *= math.sqrt(1.0 / (2.0 * n))
    return 2.0 * basis.dot(data)


def mfcc_matrix(y, sr, n_mfcc=N_MFCC):
    mel_power = mel_spectrogram(y, sr)
    log_mel = power_to_db(mel_power)
    return dct_type_2_ortho(log_mel, n_mfcc)


def polynomial_derivative_value(coefficients, x, order):
    if order >= coefficients.shape[-1]:
        return np.zeros(coefficients.shape[:-1], dtype=np.float64)
    total = np.zeros(coefficients.shape[:-1], dtype=np.float64)
    for power in range(order, coefficients.shape[-1]):
        scale = math.factorial(power) / math.factorial(power - order)
        total += coefficients[..., power] * scale * (x ** (power - order))
    return total


def savgol_delta(data, order=1, width=9):
    data = np.asarray(data, dtype=np.float64)
    frames = data.shape[-1]
    if frames < 3:
        return np.zeros_like(data)

    width = min(width, frames if frames % 2 == 1 else frames - 1)
    if width < 3:
        return np.zeros_like(data)

    half = width // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, N=order + 1, increasing=True)
    coeffs = np.linalg.pinv(design)[order] * math.factorial(order)

    result = np.empty_like(data)
    for t in range(half, frames - half):
        result[..., t] = np.tensordot(
            data[..., t - half : t + half + 1], coeffs, axes=([-1], [0])
        )

    edge_x = np.arange(width, dtype=np.float64)
    edge_design = np.vander(edge_x, N=order + 1, increasing=True)
    left_fit = data[..., :width].dot(np.linalg.pinv(edge_design).T)
    right_fit = data[..., -width:].dot(np.linalg.pinv(edge_design).T)

    for t in range(half):
        result[..., t] = polynomial_derivative_value(left_fit, float(t), order)
        right_t = width - half + t
        result[..., frames - half + t] = polynomial_derivative_value(
            right_fit, float(right_t), order
        )

    return result


def onset_strength_mean(
    y, sr, hop_length=HOP_LENGTH, n_fft=N_FFT, fmax=LOW_FREQ_CUTOFF
):
    mel_power = mel_spectrogram(
        y, sr, n_fft=n_fft, hop_length=hop_length, fmax=fmax
    )
    log_mel = power_to_db(mel_power, ref="max")
    if log_mel.shape[1] < 2:
        return 0.0

    diff = np.maximum(0.0, log_mel[:, 1:] - log_mel[:, :-1])
    pad_width = 1 + n_fft // (2 * hop_length)
    onset_env = np.pad(diff.mean(axis=0), (pad_width, 0), mode="constant")
    onset_env = onset_env[: log_mel.shape[1]]
    return float(np.mean(onset_env)) if onset_env.size else 0.0


def finite_float(value, fallback):
    value = float(value)
    return value if math.isfinite(value) else fallback
