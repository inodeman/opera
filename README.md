# opera 🎶

Kit de canto lírico sintético: 5 voces por síntesis de formantes + piano muestreado (FluidSynth) + módulo de limpieza de ruido.

## Voces
| voz | tesitura (demo) | rasgos |
|---|---|---|
| `soprano` | Si♭4–Re5 | formantes altos, sintonía F1≈f0, vibrato 5.6 Hz |
| `mezzo` | Sol4–Si4 | timbre más oscuro, más aliento |
| `tenor` | Si♭3–Re4 | formante del cantante (~3 kHz) marcado |
| `baritono` | Sol3–Si3 | formantes más bajos, cierre glotal más corto |
| `bajo` | Re3–Fa#3 | formantes más bajos, vibrato lento 5.0 Hz |

Técnica: fuente glotal Rosenberg (derivada, sobremuestreada ×4 sin aliasing), vibrato con retardo y deriva, jitter, shimmer, ruido de aliento modulado por el pulso, 5 formantes por vocal con fundido suave (legato), portamento, consonantes simples (m, v, r) y reverb de sala.

## Instalación
```bash
sudo apt install fluidsynth ffmpeg          # macOS: brew install fluid-synth ffmpeg
pip install -r requirements.txt
# SoundFont de piano (no se incluye en el repo, ~140 MB):
sudo apt install fluid-soundfont-gm         # instala /usr/share/sounds/sf2/FluidR3_GM.sf2
# o descárgalo manualmente (p. ej. https://member.keymusician.com/Member/FluidR3_GM/index.html)
# y colócalo en soundfonts/FluidR3_GM.sf2  o exporta OPERA_SF2=/ruta/FluidR3_GM.sf2
```

## Uso
```bash
python opera.py --voz soprano --pieza ave_maria --limpiar            # -> ave_maria_soprano.wav
python opera.py --voz bajo --limpiar --mp3 --mp4 -o bajo.wav         # también mp3 y mp4 (forma de onda)
python limpiar.py entrada.wav salida.wav                             # limpieza de cualquier wav
```
La limpieza aplica: paso-alto 60 Hz, compuerta espectral (`noisereduce`, con alternativa en scipy), de-esser 5–9 kHz y limitador suave (sin clipping). Imprime estadísticas (pico, clips, energía >10 kHz, rango dinámico).

## Demos
`demos/ave_maria_<voz>.mp3` y `.mp4` (~10 s, frase "A-ve Ma-ri-a" de Schubert, D.839).

`demos/ave_maria_30s_<voz>.mp3` y `.mp4` (~30 s, frase extendida: `--pieza ave_maria_30s`).

## Limitaciones
- Es síntesis paramétrica: suena claramente artificial, sobre todo en notas largas; no es una voz humana.
- Sólo vocales y consonantes rudimentarias; el texto apenas se entiende ("a-e-ma-ri-a").
- Una sola pieza incluida; añadir más es editar `PIEZAS` en `opera.py`.
- Para realismo mayor habría que usar un modelo neuronal (DiffSinger / NNSVS / OpenUtau con voicebanks), que requiere modelos preentrenados grandes y no se incluye.

Licencia: MIT
