import numpy as np
from scipy.signal import spectral
from sklearn.decomposition import PCA


def Spectral_Feat(signal):
    # Spectral Features
    [f, power_spectrum] = spectral.periodogram(signal)
    [f, Cross_spectral_density] = spectral.csd(signal, signal)
    [f, Magnitude_squared_coherence] = spectral.coherence(signal, signal)
    [f, t, Spectrogram] = spectral.spectrogram(signal)
    [f, t, Short_Time_Fourier_Transform] = spectral.stft(signal)

    # Feature Extraction
    pca = PCA(n_components=1)
    sp = pca.fit_transform(Spectrogram)
    stft = pca.fit_transform(Short_Time_Fourier_Transform.astype('float'))

    ps = power_spectrum[:Cross_spectral_density.shape[0]]
    cs = Cross_spectral_density
    ms = Magnitude_squared_coherence
    fe = np.concatenate([ps, cs, ms, sp.ravel(), stft.ravel()])

    return fe


