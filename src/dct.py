import librosa
import warnings
from librosa.filters import get_window, window_sumsquare
from librosa import util
from scipy.fftpack import dct, idct
from librosa.core.spectrum import __overlap_add
from librosa.util.exceptions import ParameterError
import numpy as np
import torch
import torch.nn.functional as F
import torch_dct

def sdct_torch(signals, n_fft, window_length, hop_length, window=torch.hamming_window, pad_mode='reflect', center=True):
    """Compute Short-Time Discrete Cosine Transform of `signals`.
    No padding is applied to the signals.
    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.
    frame_length : Window length and DCT frame length in samples.
    frame_step : Number of samples between adjacent DCT columns.
    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.
    Returns
    -------
    dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor.
    """
    if center:
        signal_dim = signals.dim()
        extended_shape = [1] * (3 - signal_dim) + list(signals.size())
        pad = int(n_fft // 2)
        signals = F.pad(signals.view(extended_shape), [pad, pad], pad_mode)
        signals = signals.view(signals.shape[-signal_dim:])
    framed = signals.unfold(-1, n_fft, hop_length)
    if callable(window):
        window = window(window_length).to(framed)
        paddings = (n_fft - window_length) // 2
        window = F.pad(window, (paddings, paddings))
    
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm="ortho").transpose(-1, -2)

def sdct_torch_1(signals, frame_length, frame_step, window=torch.hamming_window):
    """Compute Short-Time Discrete Cosine Transform of `signals`.
    No padding is applied to the signals.
    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.
    frame_length : Window length and DCT frame length in samples.
    frame_step : Number of samples between adjacent DCT columns.
    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.
    Returns
    -------
    dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor.
    """
    framed = signals.unfold(-1, frame_length, frame_step)
    if callable(window):
        window = window(frame_length).to(framed)
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm="ortho").transpose(-1, -2)


def isdct_torch(dcts, *, window_length, frame_step, frame_length=None, window=torch.hamming_window, center=True):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.
    Parameters other than `dcts` are keyword-only.
    Parameters
    ----------
    dcts : DCT matrix/matrices from `sdct_torch`
    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct_torch`).
    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_torch`.
    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.
    Returns
    -------
    signals : Time-domain signal(s) reconstructed from `dcts`, a `[..., n_samples]` tensor.
        Note that `n_samples` may be different from the original signals' lengths as passed to `sdct_torch`,
        because no padding is applied.
    """
    *_, frame_length2, n_frames = dcts.shape
    assert frame_length in {None, frame_length2}
    signals = torch_overlap_add(
        torch_dct.idct(dcts.transpose(-1, -2), norm="ortho").transpose(-1, -2),
        frame_step=frame_step,
    )
    # if callable(window):
    #     window = window(frame_length2).to(signals)
    if callable(window):
        window = window(window_length)
        # paddings = (frame_length2 - window_length) // 2
        # window = F.pad(window, (paddings, paddings))

    # if window is not None:
    #     window_frames = window[:, None].expand(-1, n_frames)
    #     window_signal = torch_overlap_add(window_frames, frame_step=frame_step)
    #     signals = signals / window_signal

    if window is not None:
        window_sum = window_sumsquare(
            window.numpy(), n_frames, hop_length=frame_step,
            win_length=window_length, n_fft=frame_length2,
            dtype=np.float32)
            
        # remove modulation effects
        approx_nonzero_indices = torch.from_numpy(
            np.where(window_sum > util.tiny(window_sum))[0])
        window_sum = torch.from_numpy(window_sum).cuda()
        signals[..., approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
        
    if center:
        pad = int(frame_length2 // 2)
        signals = signals[..., pad:-pad]
        # signals = signals[..., :-pad]
    return signals


def torch_overlap_add(framed, *, frame_step, frame_length=None):
    """Overlap-add ("deframe") a framed signal.
    Parameters other than `framed` are keyword-only.
    Parameters
    ----------
    framed : Tensor of shape `(..., frame_length, n_frames)`.
    frame_step : Overlap to use when adding frames.
    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_torch`.
    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        Tensor of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    return torch.nn.functional.fold(
        framed.reshape(-1, frame_length2, n_frames),
        output_size=(((n_frames - 1) * frame_step + frame_length2), 1),
        kernel_size=(frame_length2, 1),
        stride=(frame_step, 1),
    ).reshape(*rest, -1)


def librosa_stdct(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
):
    """Short-time Fourier transform (STFT).

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and

    - ``np.angle(D[f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.

    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        input signal

    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to ``win_length // 4`` (see below).

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:

        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        Defaults to a raised cosine window (`'hann'`), which is adequate for
        most applications in audio signal processing.

        .. see also:: `filters.get_window`

    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.

        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.

        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.

        .. see also:: `librosa.stream`

    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.

    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="reflect"``),
        ``y`` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If ``center=False``,  this argument is ignored.

        .. see also:: `numpy.pad`


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.


    See Also
    --------
    istft : Inverse STFT

    reassigned_spectrogram : Time-frequency reassigned spectrogram


    Notes
    -----
    This function caches at level 20.


    Examples
    --------

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)

    Use left-aligned frames, instead of centered frames

    >>> S_left = librosa.stft(y, center=False)


    Use a shorter hop length

    >>> D_short = librosa.stft(y, hop_length=64)


    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                        ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax)
    >>> ax.set_title('Power spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    # TODO: 這裡是處理 window length 的部分
    fft_window = get_window(window, win_length, fftbins=True)
    print(fft_window.shape)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)
    print(fft_window.shape)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))
    print(fft_window.shape)

    # Check audio is valid
    # util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too large for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    print(y_frames.shape)

    if dtype is None:
        dtype = y.dtype
        # dtype = util.dtype_r2c(y.dtype)


    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (n_fft, y_frames.shape[1]), dtype=dtype, order="F"
    )

    fft = dct

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        tmp = fft_window * y_frames[:, bl_s:bl_t]
        print(tmp.shape)
        stft_matrix[:, bl_s:bl_t] = fft(
            fft_window * y_frames[:, bl_s:bl_t], norm="ortho")
            
    return stft_matrix

def librosa_istdct(
    stft_matrix,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    length=None,
):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``
    by minimizing the mean squared error between ``stft_matrix`` and STFT of
    ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified ``stft_matrix``.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from ``stft``

    hop_length : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to ``win_length // 4``.

    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the ``window`` function (see below).

        If unspecified, defaults to ``n_fft``.

    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, ``D`` is assumed to have centered frames.
        - If ``False``, ``D`` is assumed to have left-aligned frames.

    dtype : numeric type
        Real numeric type for ``y``.  Default is to match the numerical
        precision of the input spectrogram.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from ``stft_matrix``

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([-1.407e-03, -4.461e-04, ...,  5.131e-06, -1.417e-05],
          dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of ``y`` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    8.940697e-08
    """

    # n_fft = 2 * (stft_matrix.shape[0] - 1)
    n_fft = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = util.pad_center(ifft_window, n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if dtype is None:
        # dtype = util.dtype_c2r(stft_matrix.dtype)
        dtype = stft_matrix.dtype

    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    fft = idct
    import numba
    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft(stft_matrix[:, bl_s:bl_t], norm='ortho')

        print(numba.typeof(ytmp), numba.typeof(y))

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window,
        n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2) : -int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = util.fix_length(y[start:], length)

    return y


if __name__ == '__main__':
    import timeit, numba
    example_audio = librosa.core.load(
        librosa.util.example_audio_file(), offset=30, duration=5
    )[0]

    start_time = timeit.default_timer()
    # sdct = stdct(example_audio[:32000], 512,100,400)
    sdct = sdct_torch(torch.tensor(example_audio[:32000]), 512 ,400 ,100, window=torch.hann_window)
    # sdct = sdct_torch_1(torch.tensor(example_audio[:32000]), 512 ,100, window=torch.hann_window)
    print(timeit.default_timer() - start_time)
    print(sdct.shape)
    # print(numba.typeof(sdct))

    start_time = timeit.default_timer()
    # stft = torch.stft(torch.tensor(example_audio[:32000]), 512,100,400)
    stft = torch.stft(torch.tensor(example_audio[:32000]), 512,100,512)
    print(timeit.default_timer() - start_time)
    print(stft.shape)
    isdct = isdct_torch(sdct, window_length=400, frame_step=100, window=torch.hann_window)
    # print(sum(torch.tensor(example_audio[:32000]-sdct)))
    # sdct = stdct(example_audio[:32000], 512,100,400)
    # print(sdct.shape)