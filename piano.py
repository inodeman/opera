"""Piano muestreado vía fluidsynth + SoundFont GM (FluidR3_GM)."""
import os, struct, subprocess, tempfile, numpy as np, soundfile as sf
SF_RUTAS = [os.environ.get('OPERA_SF2', ''), 'soundfonts/FluidR3_GM.sf2', '/usr/share/sounds/sf2/FluidR3_GM.sf2']

def _vlq(n):
    b = [n & 127]; n >>= 7
    while n: b.insert(0, (n & 127) | 128); n >>= 7
    return bytes(b)

def piano(eventos, L, sr=44100, prog=0):
    """eventos: (inicio_s, dur_s, midi, velocidad)."""
    sf2 = next((p for p in SF_RUTAS if p and os.path.exists(p)), None)
    if not sf2: raise FileNotFoundError('No se encontró FluidR3_GM.sf2 (ver README / OPERA_SF2)')
    ev = []
    for s, d, n, v in eventos: ev += [(int(s*960), 0x90, n, v), (int((s+d)*960), 0x80, n, 0)]
    ev.sort(); trk = bytes([0, 0xC0, prog]); last = 0
    for tk, st, n, v in ev: trk += _vlq(tk-last)+bytes([st, n, v]); last = tk
    trk += b'\x00\xff\x2f\x00'
    d = tempfile.mkdtemp(); mid, wav = f'{d}/p.mid', f'{d}/p.wav'
    open(mid, 'wb').write(b'MThd'+struct.pack('>IHHH', 6, 0, 1, 480)+b'MTrk'+struct.pack('>I', len(trk))+trk)
    subprocess.run(['fluidsynth', '-ni', '-g', '0.6', '-R', '0', '-C', '0', '-r', str(sr), '-F', wav, sf2, mid], check=True, capture_output=True)
    x, _ = sf.read(wav); x = x.mean(1) if x.ndim > 1 else x
    x = np.pad(x, (0, max(0, L-len(x))))[:L]; return x/np.abs(x).max()
