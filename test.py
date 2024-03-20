import matplotlib.pyplot as plt
import librosa
import numpy as np

y, sr = librosa.load("./p225xp225.wav")

mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis="mel", fmax=8000, x_axis="time")
plt.title("Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.savefig("../private/spec.png")
