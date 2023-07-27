
from scipy.signal import butter, filtfilt
from config import cfg
import wfdb
import numpy as np
import scipy.signal

import pyedflib





def get_namelist():
    namelist =  [ "r01.edf", "r04.edf", "r07.edf","r08.edf", "r10.edf"]
    return namelist

def get_data(name, channel = [0,1,2,3]):
    path = cfg().DB + '/'
    fs = 1000
    f = pyedflib.EdfReader(path + name)
    ecg = []
    for i in range(len(channel)):
        ecg_tmp = f.readSignal(channel[i]+1)
        ecg.append(ecg_tmp)
    ecg = np.array(ecg)
    ecg = ecg.reshape((len(channel),len(ecg[0])))
  
    signal_annotation = wfdb.rdann(path + name, "qrs", sampfrom=0, sampto=60000*5)
 
    peaks = signal_annotation.sample

    return ecg, peaks


def pad_audio(audio, segment_samples):
    r"""Pad the audio with zero in the end so that the length of audio can
    be evenly divided by segment_samples.

    Args:
        audio: (channels_num, audio_samples)

    Returns:
        padded_audio: (channels_num, audio_samples)
    """
    channels_num, audio_samples = audio.shape

    # Number of segments
    segments_num = int(np.ceil(audio_samples / segment_samples))

    pad_samples = segments_num * segment_samples - audio_samples

    padded_audio = np.concatenate(
        (audio, np.zeros((channels_num, pad_samples))), axis=1
    )
    # (channels_num, padded_audio_samples)

    return padded_audio


def enframe(sig, segment_samples):
    audio_samples = sig.shape[1]
    hop_samples =  segment_samples // 2
    
    segments = []
    
    pointer = 0
    while pointer + segment_samples <= audio_samples:
        segments.append(sig[:, pointer : pointer + segment_samples])
        pointer += hop_samples

    segments = np.array(segments)
    
    return segments

def deframe(segments):
    def _is_integer(x: float) -> bool:
        if x - int(x) < 1e-10:
            return True
        else:
            return False

    
    (segments_num, _, segment_samples) = segments.shape

    if segments_num == 1:
        return segments[0]

    assert _is_integer(segment_samples * 0.25)
    assert _is_integer(segment_samples * 0.75)

    output = []

    output.append(segments[0, :, 0 : int(segment_samples * 0.75)])

    for i in range(1, segments_num - 1):
        output.append(
            segments[
                i, :, int(segment_samples * 0.25) : int(segment_samples * 0.75)
            ]
        )

    output.append(segments[-1, :, int(segment_samples * 0.25) :])

    output = np.concatenate(output, axis=-1)
    output = output.flatten()
    return output