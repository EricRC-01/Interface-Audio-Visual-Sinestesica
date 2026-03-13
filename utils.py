

import numpy as np # Uso de formululas matimaticas mais complexas
import soundfile as sf  # captação de audio 
import librosa  # ler spectrograma
import librosa.display # plotar imagem
import aubio    # leitura de BPM 
import matplotlib.pyplot as plt # plotar imagem









def processar_espectograma(vet_ampli):
    # Aplicação do STFT (Short-Time Fourier Transform) no vetor de amplitudes, 
    # para converter o sinal do domínio do tempo para o domínio da frequência. 
    # O resultado é um espectro complexo que contém informações de amplitude e fase. 
    Espectograma = librosa.stft(vet_ampli)

    # np.abs(Espectograma) → pega apenas a magnitude (descarta fase)
    magnitude = np.abs(Espectograma)
    
    # amplitude_to_db → converte amplitude para decibéis
    # ref=np.max → normaliza pelo maior valor (0 dB = pico)
    Vet_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    return Vet_db, Espectograma
