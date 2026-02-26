

import numpy as np # Uso de formululas matimaticas mais complexas
import soundfile as sf  # captação de audio 
import librosa  # ler spectrograma
import librosa.display # plotar imagem
import aubio    # leitura de BPM 
import matplotlib.pyplot as plt # plotar imagem


# Coletar audio
# sample_rate = taxa de divisão de amostras de audio 
# vet_ampli = vetor de amplitudes
audio_path = "Ozzy Osbourne - No More Tears.mp3"
vet_ampli, sample_rate = librosa.load(audio_path, sr=None, duration=60.0)  # Carrega os primeiros 60 segundos do áudio

# Verificar as dimensões do vetor de amplitudes e a taxa de amostragem
print(f"vet_ampli.shape - {vet_ampli.shape}")
print(f"sample_rate - {sample_rate} Hz")


# Aplicação do STFT (Short-Time Fourier Transform) no vetor de amplitudes, 
# para converter o sinal do domínio do tempo para o domínio da frequência. 
# O resultado é um espectro complexo que contém informações de amplitude e fase. 
D = librosa.stft(vet_ampli)


# np.abs(D) → pega apenas a magnitude (descarta fase)
# amplitude_to_db → converte amplitude para decibéis
# ref=np.max → normaliza pelo maior valor (0 dB = pico)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plotar o espectrograma
plt.figure(figsize=(10, 4))

# Mostra o espectrograma como imagem
# x_axis="time" → eixo X em segundos
# y_axis="hz" → eixo Y em frequência (Hz)
# sr=sr → usa o sample rate extraido anteriormente 
librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="hz")
plt.colorbar(format="%+2.0f dB") # adiciona a barra de corres e mostra a intensidade em dB
plt.title("Espectrograma") # Define o título do gráfico

# ajustes de layout e exibição
plt.tight_layout()
plt.show()

# Estimativa de BPM usando librosa
tempo, beats = librosa.beat.beat_track(y=vet_ampli, sr=sample_rate)
print(f"Estimated tempo: {tempo[0]:.2f} BPM")



# definindo a harmonia (chroma)
chroma = librosa.feature.chroma_stft(y=vet_ampli, sr=sample_rate)

# Plotando a harmonia (chroma)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", sr=sample_rate)
plt.colorbar()
plt.title("Chroma (Harmonia)")
plt.tight_layout()
plt.show()






