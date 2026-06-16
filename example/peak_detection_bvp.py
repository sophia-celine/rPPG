import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =====================================================
# Simulação de um sinal de BVP
# =====================================================

fs = 30  # frequência de amostragem (Hz)
duracao = 8  # segundos

t = np.arange(0, duracao, 1/fs)

# Frequência cardíaca simulada (~75 bpm)
fc = 75 / 60  # Hz

# Forma de onda básica "sobe e desce" (senoidal simples, sem nó dicrotico)
bvp = np.sin(2*np.pi*fc*t)

# Adiciona leve flutuação de amplitude (Modulação de Amplitude)
bvp *= (1.0 + 0.1 * np.sin(2*np.pi*0.1*t))

# Adiciona leve flutuação de linha de base (Baseline Wander)
bvp += 0.08 * np.sin(2*np.pi*0.15*t)

# Adiciona ruído muito reduzido
np.random.seed(42)
bvp += 0.005*np.random.randn(len(t))

# =====================================================
# Detecção de picos
# =====================================================

peaks, _ = find_peaks(
    bvp,
    distance=int(fs*0.5),  # mínimo de 0.5 s entre picos
    prominence=0.5
)

# =====================================================
# Plotagem
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(t, bvp, lw=2, label='BVP')
plt.scatter(
    t[peaks],
    bvp[peaks],
    color='red',
    s=60,
    zorder=3,
    label='Pico detectado'
)

# =====================================================
# Anotação do intervalo entre dois picos
# =====================================================

if len(peaks) >= 2:

    p1 = peaks[0]
    p2 = peaks[1]

    y_annot = max(bvp)

    plt.plot(
        [t[p1], t[p2]],
        [y_annot, y_annot],
        '--',
        color='green',
        lw=2
    )

    plt.plot(
        [t[p1], t[p1]],
        [y_annot-0.03, y_annot+0.03],
        color='green',
        lw=2
    )

    plt.plot(
        [t[p2], t[p2]],
        [y_annot-0.03, y_annot+0.03],
        color='green',
        lw=2
    )

    plt.text(
        (t[p1] + t[p2]) / 2,
        y_annot + 0.08,
        'Intervalo entre picos',
        color='green',
        ha='center',
        fontsize=12
    )

plt.xlabel('Tempo (s)', fontsize='large')
plt.ylabel('Amplitude', fontsize='large')

# Aumenta o tamanho dos números nos eixos
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize='large')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()