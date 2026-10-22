#!/usr/bin/env python3
"""CLI: python opera.py --voz soprano --pieza ave_maria --limpiar [-o salida.wav] [--mp3] [--mp4]"""
import argparse, subprocess, numpy as np, scipy.signal as ss, soundfile as sf
from voces import cantar, VOCES, SR
from piano import piano
from limpiar import limpiar, estadisticas

PIEZAS = {
 # Schubert, Ave Maria D.839 (en Si bemol): "A - ve Ma - ri - a"
 'ave_maria': dict(
   notas=[(70,1.9,'a'),(69,.35,'ve'),(70,.85,'e'),(74,.8,'ma'),(72,.35,'a'),(70,.9,'ri'),(70,2.6,'a')],
   acordes=['Bb','Bb','Eb','Bb','F7','F7','Bb','Bb','Bb','Bb','Bb','Bb'], pulso=.9,
   arp={'Bb':[46,53,58,62,65,62],'Eb':[51,55,58,63,67,63],'F7':[41,53,57,60,63,60]}, dur=10.5),
}

PIEZAS['ave_maria_30s'] = dict(
   # "A-ve Ma-ri-a, gra-ti-a ple-na, Ma-ri-a gra-ti-a ple-na, Do-mi-nus te-cum" (aprox., Si bemol)
   notas=PIEZAS['ave_maria']['notas'][:-1]+[(70,1.6,'a'),
     (72,.5,'ra'),(74,.4,'ti'),(75,.5,'a'),(74,1.0,'e'),(72,.4,'ne'),(70,1.6,'a'),
     (74,.9,'ma'),(77,.5,'ri'),(75,.4,'a'),(74,.9,'a'),(72,.5,'ra'),(70,.4,'ti'),(69,.5,'a'),(70,.8,'e'),(72,.4,'na'),(74,1.6,'a'),
     (77,.9,'o'),(75,.5,'mi'),(74,.5,'nu'),(72,.9,'e'),(70,.5,'u'),(69,.5,'e'),(70,3.0,'u')],
   acordes=['Bb','Bb','Eb','Bb','F7','F7','Bb','Bb','Bb','F7','Bb','F7','F7','Bb','Bb',
            'Bb','Eb','Eb','Bb','F7','F7','Bb','Bb','Eb','Bb','F7','F7','Bb','Bb','Bb','Bb','Bb','Bb','Bb'],
   pulso=.9, arp=PIEZAS['ave_maria']['arp'], dur=30.5)

def reverb(x, t60=2.0, wet=.2):
    n = int(t60*SR); r = np.random.default_rng(1)
    ir = ss.sosfilt(ss.butter(2, 4500, fs=SR, output='sos'), r.standard_normal(n))*np.exp(-6.9*np.arange(n)/n)
    ir[:int(.025*SR)] = 0; w = ss.fftconvolve(x, ir)[:len(x)]
    return x + wet*w*np.abs(x).max()/np.abs(w).max()

def render(voz, pieza='ave_maria', limpio=True):
    P = PIEZAS[pieza]; v = cantar(P['notas'], voz); L = len(v)
    tr = VOCES[voz]['tr']; sh = -12 if tr <= -12 else 0   # piano transpuesto a la tonalidad de la voz
    ev = []; b = P['pulso']
    for i, c in enumerate(P['acordes']):
        for j, n in enumerate(P['arp'][c]):
            ev.append((i*b+j*b/6, b*.9, n+tr-sh, 48 if j else 66))
    p = piano(ev, L)
    mix = reverb(v*.8+p*.42)[:int(P['dur']*SR)]
    fo = int(.8*SR); mix[-fo:] *= np.linspace(1, 0, fo)**2; mix = mix/np.abs(mix).max()*.85
    if limpio: mix = limpiar(mix, SR, reduccion=.5)
    mix = mix/np.abs(mix).max()*.89
    return mix

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--voz', choices=VOCES, default='soprano')
    a.add_argument('--pieza', choices=PIEZAS, default='ave_maria'); a.add_argument('--limpiar', action='store_true')
    a.add_argument('-o', '--salida'); a.add_argument('--mp3', action='store_true'); a.add_argument('--mp4', action='store_true')
    g = a.parse_args(); out = g.salida or f'{g.pieza}_{g.voz}.wav'
    y = render(g.voz, g.pieza, g.limpiar); sf.write(out, y, SR, subtype='PCM_16')
    print(out, estadisticas(y, SR)); base = out.rsplit('.', 1)[0]
    if g.mp3 or g.mp4:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out, '-b:a', '160k', base+'.mp3'], check=True)
    if g.mp4:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', base+'.mp3', '-filter_complex',
          f"[0:a]showwavespic=s=1280x720:colors=gold,drawtext=text='{g.voz.capitalize()} - Ave Maria (Schubert)':fontcolor=white:fontsize=48:x=(w-tw)/2:y=40[v]",
          '-map', '[v]', '-map', '0:a', '-c:v', 'libx264', '-tune', 'stillimage', '-pix_fmt', 'yuv420p', '-r', '2', '-c:a', 'aac', '-b:a', '160k', '-shortest', base+'.mp4'], check=True)
