import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter

# ===============================
# Utility functions
# ===============================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def soft_saturation(x, drive=2.0):
    return np.tanh(drive * x)

# ===============================
# Wow & Flutter
# ===============================

def wow_flutter(audio, fs, wow_depth=0.002, wow_rate=0.3, flutter_depth=0.0005, flutter_rate=6.0):
    """
    Applies pitch modulation using modulated delay resampling.
    """
    t = np.arange(len(audio)) / fs
    # Combined modulation
    mod = (
        wow_depth * np.sin(2 * np.pi * wow_rate * t)
        + flutter_depth * np.sin(2 * np.pi * flutter_rate * t + np.random.rand() * 2*np.pi)
    )

    # Add slow random jitter to avoid perfect periodicity
    mod += np.random.normal(0, 0.0002, size=len(audio))

    # Compute resampling factor
    indices = np.arange(len(audio)) * (1 + mod)
    indices = np.clip(indices, 0, len(audio) - 1)

    # Linear interpolation resampling
    i0 = np.floor(indices).astype(int)
    i1 = np.clip(i0 + 1, 0, len(audio) - 1)
    frac = indices - i0
    return audio[i0] * (1 - frac) + audio[i1] * frac

# ===============================
# Failure (dropouts / glitches)
# ===============================

def failure(audio, fs, probability=0.003, dropout_dur=0.05):
    """
    Randomly mutes or distorts small sections.
    """
    out = audio.copy()
    n = len(audio)
    num_dropouts = int(probability * n)
    for _ in range(num_dropouts):
        start = np.random.randint(0, n - int(dropout_dur * fs))
        end = start + int(dropout_dur * fs)
        # Apply a random failure type
        r = np.random.rand()
        if r < 0.5:
            out[start:end] *= np.random.uniform(0.0, 0.3)  # Volume dropout
        else:
            # Reverse section (tracking error)
            out[start:end] = out[start:end][::-1]
    return out

# ===============================
# Noise & EQ
# ===============================

def add_tape_noise(audio, fs, noise_level=0.01):
    # Generate pinkish noise
    noise = np.random.randn(len(audio))
    noise = bandpass_filter(noise, 300, 8000, fs)
    noise /= np.max(np.abs(noise) + 1e-6)
    return audio + noise_level * noise

def lo_fi_eq(audio, fs):
    # Band-limit to simulate VHS/tape
    return bandpass_filter(audio, 100, 10000, fs)

# ===============================
# Main Processing Chain
# ===============================

def process_file(path_in, path_out, drive=2.5):
    audio, fs = librosa.load(path_in, sr=None, mono=True)

    # Apply Wow & Flutter
    wf = wow_flutter(audio, fs)

    # Apply Failure‚
    fail = failure(wf, fs)

    # Apply Saturation
    sat = soft_saturation(fail, drive=drive)

    # Add Noise
    noisy = add_tape_noise(sat, fs, noise_level=0.015)

    # EQ shaping
    out = lo_fi_eq(noisy, fs)

    # Normalize and save
    out /= np.max(np.abs(out) + 1e-6)
    sf.write(path_out, out, fs)
    print(f"Processed file saved to: {path_out}")

# ===============================
# Run
# ===============================

if __name__ == "__main__":
    process_file("synth‚.wav", "output_genloss.wav")
