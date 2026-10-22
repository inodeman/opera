"""Síntesis de voz cantada por formantes (fuente glotal Rosenberg/LF, vibrato, jitter, shimmer, aliento, legato)."""
import numpy as np, scipy.signal as ss
SR = 44100
OS = 4  # sobremuestreo de la fuente glotal (anti-aliasing)

# Formantes base (voz masculina adulta): vocal -> [(Hz, ancho de banda, dB)]
FORM = {
 'a': [(750,80,0),(1150,90,-5),(2600,120,-18),(3000,130,-20),(3500,150,-30)],
 'e': [(450,70,0),(1800,90,-11),(2550,110,-16),(2950,120,-20),(3500,150,-30)],
 'i': [(300,50,0),(2150,100,-16),(2750,110,-14),(3100,120,-18),(3600,150,-30)],
 'o': [(470,70,0),(800,80,-8),(2600,110,-20),(3000,130,-24),(3500,150,-34)],
 'u': [(330,50,0),(700,60,-12),(2500,150,-28),(3000,150,-30),(3500,180,-40)],
 'm': [(250,60,0),(1100,150,-22),(2200,200,-30),(2900,200,-34),(3500,200,-40)],  # nasal
}

VOCES = {
 # escala formantes, transposición (semitonos vs soprano), vibrato Hz, profundidad (fracción), aliento, OQ, formante del cantante dB
 'soprano':  dict(fs=1.18, tr=0,   vr=5.6, vd=.018, br=.012, oq=.65, sf=-6,  f1track=True),
 'mezzo':    dict(fs=1.10, tr=-3,  vr=5.3, vd=.020, br=.015, oq=.60, sf=-4,  f1track=True),
 'tenor':    dict(fs=1.00, tr=-12, vr=5.8, vd=.016, br=.010, oq=.55, sf=0,   f1track=False),
 'baritono': dict(fs=0.94, tr=-15, vr=5.4, vd=.017, br=.010, oq=.50, sf=1,   f1track=False),
 'bajo':     dict(fs=0.88, tr=-20, vr=5.0, vd=.018, br=.008, oq=.45, sf=2,   f1track=False),
}

def midi2hz(n): return 440*2**((n-69)/12)
def _lp(x, fc, o=2, sr=SR): return ss.filtfilt(*ss.butter(o, fc/(sr/2)), x)

def _smooth_noise(L, fc, rng, sr=SR):
    n = rng.standard_normal(L); n = _lp(n, fc, 2, sr); return n/ (n.std()+1e-12)

def glottal(f0, oq, rng):
    """Derivada de pulso Rosenberg con fase continua, a SR*OS y decimada."""
    f = np.repeat(f0, OS); ph = np.cumsum(f)/(SR*OS) % 1.0
    tp, tn = oq*0.66, oq*0.34  # apertura / cierre
    g = np.where(ph < tp, 0.5*(1-np.cos(np.pi*ph/tp)),
        np.where(ph < tp+tn, np.cos(np.pi/2*(ph-tp)/tn), 0.0))
    d = np.diff(g, prepend=0)                # flujo -> derivada (radiación labial)
    d = ss.resample_poly(d, 1, OS)[:len(f0)]
    return d/ (np.abs(d).max()+1e-12)

def _reson(x, fc, bw):
    r = np.exp(-np.pi*bw/SR); th = 2*np.pi*fc/SR
    return ss.lfilter([1-r], [1, -2*r*np.cos(th), r*r], x)

def cantar(notas, voz='soprano', cola=1.5, semilla=0):
    """notas: lista de (midi_soprano, dur_s, silaba) — silaba: vocal con consonante opcional delante ('ve','ma','ri','a')."""
    P = VOCES[voz]; rng = np.random.default_rng(semilla)
    L = int((sum(d for _, d, _ in notas)+cola)*SR); t = np.arange(L)/SR
    f0 = np.zeros(L); amp = np.zeros(L); onset = np.zeros(L); fric = np.zeros(L)
    W = {k: np.zeros(L) for k in FORM}; tt = 0.0
    for n, d, s in notas:
        a, b = int(tt*SR), int((tt+d)*SR); v = s[-1]; c = s[:-1]
        f0[a:b] = midi2hz(n+P['tr']); amp[a:b] = 1; W[v][a:b] = 1
        if c:
            onset[a:b] = t[a:b]-tt
            k = a+int(.07*SR)
            if c == 'm': W[v][a:k] = 0; W['m'][a:k] = 1; amp[a:k] = .55
            elif c == 'v': amp[a:a+int(.05*SR)] = .3; fric[a:a+int(.05*SR)] = 1
            elif c == 'r': amp[a:a+int(.025*SR)] = .35
        elif tt == 0: onset[a:b] = t[a:b]
        else: onset[a:b] = onset[a-1]+t[a:b]-t[a-1]  # misma vocal: legato, vibrato continúa
        tt += d
    e = int(tt*SR); f0[e:] = f0[e-1]; W[notas[-1][2][-1]][e:] = 1; onset[e:] = onset[e-1]+t[e:]-t[e-1]
    fade = int(.7*SR); amp[e:e+fade] = np.linspace(1, 0, fade)**1.5
    # legato: portamento suave y envolventes
    f0 = np.exp(_lp(np.log(f0+1e-9), 7)); amp = np.clip(_lp(amp, 12), 0, None)
    amp[:int(.12*SR)] *= np.linspace(0, 1, int(.12*SR))**2
    for k in W: W[k] = np.clip(_lp(W[k], 14), 0, 1)
    ws = sum(W.values())+1e-9
    for k in W: W[k] /= ws
    # vibrato natural (retrasado, con deriva de frecuencia) + jitter + shimmer
    vr = P['vr']*(1+.04*_smooth_noise(L, 1, rng))
    vib = np.sin(2*np.pi*np.cumsum(vr)/SR)*np.clip((onset-.3)/.5, 0, 1)
    jit = .004*_smooth_noise(L, 30, rng) + .003*_smooth_noise(L, 3, rng)
    f = f0*(1+P['vd']*vib+jit)
    shim = 1+.05*_smooth_noise(L, 25, rng)+.03*vib
    src = glottal(f, P['oq'], rng)*shim
    # aliento: ruido modulado por el pulso glotal, filtrado
    br = _lp(rng.standard_normal(L), 6000)*(0.4+0.6*(np.abs(src) > .1))
    src = src + P['br']*br/np.abs(br).max()*8 + .15*fric*_lp(rng.standard_normal(L),5000)
    y = np.zeros(L)
    for k, fl in FORM.items():
        if W[k].max() < 1e-3: continue
        yk = np.zeros(L)
        for i, (fc, bw, g) in enumerate(fl):
            fc2 = fc*P['fs'] if i < 2 else fc*(1+(P['fs']-1)*.5)
            if i in (2, 3): g += P['sf']                       # formante del cantante (~3 kHz)
            if i == 0 and P['f1track']:                        # sintonía F1 ~ f0 (sopranos agudas)
                fc2 = max(fc2, float(np.median(f0))*1.05)
            yk += 10**(g/20)*_reson(src, fc2, bw*(1+.3*(P['fs']-1)))
        y += W[k]*yk
    y *= amp
    y = ss.sosfilt(ss.butter(4, [70, 9000], 'band', fs=SR, output='sos'), y)
    return y/np.abs(y).max()
