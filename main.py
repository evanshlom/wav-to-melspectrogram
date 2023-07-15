import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# params
n_fft = 2048
hop = int(n_fft/4)

# input
song = 'wavs\Beached.wav'
y, sr = librosa.load(song)

######################################## Signal ########################################
plt.plot(y)
plt.title('Signal from Beached.wav')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.savefig('output/signal-beached.png')
plt.clf()

######################################## Spectrum ########################################
ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))
plt.plot(ft)
plt.title('Spectrum from Beached.wav')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')
plt.savefig('output/spectrum-beached.png')
plt.clf()

######################################## Spectrogram ########################################
spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
    window='hann',
    pad_mode='reflect')
    )** 2 # square of complex magnitude of STFT
spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

librosa.display.specshow(spectrogram, sr=sr, y_axis='log', x_axis='time', hop_length=hop)
plt.title('Spectrogram from Beached.wav')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig('output/spectrogram-beached.png')
plt.clf()

######################################## Mel Spectrogram ########################################
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
    window='hann',
    pad_mode='reflect',
    power=2.0,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm='slaney',
    dtype=np.float32)
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

librosa.display.specshow(mel_spec, sr=sr, y_axis='mel', x_axis='time', hop_length=hop)
plt.title('Mel Spectrogram from Beached.wav')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig('output/mel-spectrogram-beached.png')
plt.clf()