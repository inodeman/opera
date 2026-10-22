"""Módulo de limpieza de ruido: paso-alto, compuerta espectral (noisereduce), de-esser y limitador suave.
Uso independiente: python limpiar.py entrada.wav salida.wav"""
import sys, numpy as np, scipy.signal as ss, soundfile as sf

def limpiar(x, sr, hp=60, reduccion=0.6, deess=True, lp=None):
    mono = x.ndim == 1; x = x[:, None] if mono else x
    out = []
    for ch in x.T:
        ch = ss.sosfiltfilt(ss.butter(4, hp, 'high', fs=sr, output='sos'), ch)
        try:
            import noisereduce as nr
            ch = nr.reduce_noise(y=ch, sr=sr, stationary=True, prop_decrease=reduccion)
        except ImportError:  # compuerta espectral simple
            f, t, Z = ss.stft(ch, sr, nperseg=2048); mag = np.abs(Z)
            floor = np.percentile(mag, 10, axis=1, keepdims=True)*1.5
            g = np.clip((mag-floor)/(mag+1e-12), 1-reduccion, 1)
            ch = ss.istft(Z*g, sr, nperseg=2048)[1][:len(ch)]
        if deess:  # compresión dinámica de 5-9 kHz
            band = ss.sosfiltfilt(ss.butter(2, [5000, 9000], 'band', fs=sr, output='sos'), ch)
            env = ss.sosfiltfilt(ss.butter(1, 30, fs=sr, output='sos'), np.abs(band))
            thr = np.percentile(env, 90); gain = np.minimum(1, (thr/(env+1e-12))**.5)
            ch = ch - band*(1-gain)
        if lp: ch = ss.sosfiltfilt(ss.butter(4, lp, fs=sr, output='sos'), ch)
        out.append(ch)
    y = np.stack(out, 1); pk = np.abs(y).max()
    if pk > .89: y = np.tanh(y/pk*1.2)/np.tanh(1.2)*.89   # limitador suave, sin clipping
    return y[:, 0] if mono else y

def estadisticas(x, sr):
    x = x if x.ndim == 1 else x.mean(1)
    f, P = ss.welch(x, sr, nperseg=4096)
    hiss = 10*np.log10(P[f > 10000].sum()/P.sum()+1e-20)
    fr = np.lib.stride_tricks.sliding_window_view(x, 2048)[::1024]
    rms = np.sqrt((fr**2).mean(1)); quiet = np.percentile(rms, 5); loud = np.percentile(rms, 95)
    return dict(pico=float(np.abs(x).max()), clips=int((np.abs(x) > .999).sum()),
                hiss_dB_sobre_10k=round(float(hiss), 1), rango_dinamico_dB=round(float(20*np.log10(loud/(quiet+1e-9))), 1))

if __name__ == '__main__':
    x, sr = sf.read(sys.argv[1]); y = limpiar(x, sr); sf.write(sys.argv[2], y, sr)
    print('antes', estadisticas(x, sr)); print('después', estadisticas(y, sr))
