import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter

# =====================================
# Utility functions
# =====================================

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def soft_saturation(x, drive=1.1):
    return np.tanh(drive * x)

# =====================================
# Wow & Flutter (very subtle)
# =====================================

def wow_flutter(audio, fs, wow_depth=0.0002, wow_rate=0.25, flutter_depth=0.00005, flutter_rate=5.0):
    t = np.arange(len(audio)) / fs
    mod = (
        wow_depth * np.sin(2 * np.pi * wow_rate * t)
        + flutter_depth * np.sin(2 * np.pi * flutter_rate * t)
    )
    mod = np.cumsum(mod)  # smoother pitch drift
    indices = np.linspace(0, len(audio) - 1, len(audio))
    warped = np.interp(indices + mod * fs, np.arange(len(audio)), audio, left=0, right=0)
    return warped

# =====================================
# Failure (dropouts - almost inaudible)
# =====================================

def failure(audio, fs, probability=0.00002, dropout_dur=0.008):
    out = audio.copy()
    n = len(audio)
    num_dropouts = int(probability * n)
    for _ in range(num_dropouts):
        start = np.random.randint(0, n - int(dropout_dur * fs))
        end = start + int(dropout_dur * fs)
        fade = np.linspace(1.0, 0.8, end - start)
        out[start:end] *= fade
    return out

# =====================================
# Gentle Tape Noise
# =====================================

def add_tape_noise(audio, fs, noise_level=0.00005):
    n = len(audio)
    white = np.random.randn(n)
    b, a = butter(1, [200 / (0.5 * fs), 8000 / (0.5 * fs)], btype='band')
    noise = lfilter(b, a, white)
    noise /= np.max(np.abs(noise) + 1e-6)
    return audio + noise_level * noise

# =====================================
# Light EQ shaping
# =====================================

def lo_fi_eq(audio, fs):
    return bandpass_filter(audio, 80, 12000, fs)

# =====================================
# Main Processing Chain
# =====================================

def process_file(path_in, path_out):
    audio, fs = librosa.load(path_in, sr=None, mono=True)

    wf = wow_flutter(audio, fs)
    fail = failure(wf, fs)
    sat = soft_saturation(fail, drive=1.1)
    noisy = add_tape_noise(sat, fs)
    out = lo_fi_eq(noisy, fs)

    out /= np.max(np.abs(out) + 1e-6)
    sf.write(path_out, out, fs)
    print(f"Processed file saved to: {path_out}")

# =====================================
# Run
# =====================================

if __name__ == "__main__":
    process_file("synth.wav", "output_genloss_subtle.wav")
