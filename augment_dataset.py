from pathlib import Path
import random
import shutil

import numpy as np
import soundfile as sf

from config import DATASET_DIR, SAMPLE_RATE, COMMANDS


# =========================================================
# CONFIGURACIÓN
# =========================================================

SOURCE_DATASET_DIR = Path("dataset_domotica_voz")
AUGMENTED_DIR = Path("dataset_domotica_voz_augmented")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# TÉCNICA 1: TIME SHIFTING
# =========================================================

def time_shift(audio, max_shift_fraction=0.15):
    """
    Desplaza el audio hacia adelante o atrás.
    Simula que la persona empezó a hablar un poco antes o después.
    """
    max_shift = int(len(audio) * max_shift_fraction)

    if max_shift <= 0:
        return audio

    shift = random.randint(-max_shift, max_shift)

    return np.roll(audio, shift)


# =========================================================
# TÉCNICA 2: RUIDO GAUSSIANO
# =========================================================

def add_gaussian_noise(audio, noise_factor=0.008):
    """
    Agrega ruido gaussiano leve.
    Ayuda a que el modelo sea más robusto ante ruido ambiental.
    """
    noise = np.random.normal(0, noise_factor, audio.shape)
    audio_noisy = audio + noise

    return np.clip(audio_noisy, -1.0, 1.0)


# =========================================================
# TÉCNICA 3: TIME STRETCHING SIMPLE
# =========================================================

def time_stretch_simple(audio, rate=1.1):
    """
    Cambia levemente la velocidad del audio usando interpolación.
    rate > 1.0: audio más rápido
    rate < 1.0: audio más lento

    Nota: esta versión simple no preserva perfectamente el pitch,
    pero es suficiente para aumentar variabilidad en un proyecto académico.
    """
    if rate <= 0:
        return audio

    original_indices = np.arange(len(audio))
    new_length = int(len(audio) / rate)

    if new_length <= 1:
        return audio

    new_indices = np.linspace(0, len(audio) - 1, new_length)
    stretched = np.interp(new_indices, original_indices, audio)

    # Ajustar a la longitud original
    if len(stretched) < len(audio):
        padding = len(audio) - len(stretched)
        stretched = np.pad(stretched, (0, padding), mode="constant")
    else:
        stretched = stretched[:len(audio)]

    return stretched.astype(np.float32)


# =========================================================
# OPCIONAL: CAMBIO LEVE DE VOLUMEN
# =========================================================

def change_volume(audio, factor=1.1):
    """
    Cambia ligeramente el volumen.
    No cuenta como una de las técnicas principales del enunciado,
    pero ayuda a simular diferentes distancias al micrófono.
    """
    audio_vol = audio * factor
    return np.clip(audio_vol, -1.0, 1.0)


# =========================================================
# UTILIDADES
# =========================================================

def normalize_audio(audio):
    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio.astype(np.float32)


def preparar_carpeta_salida():
    if AUGMENTED_DIR.exists():
        print(f"Eliminando carpeta anterior: {AUGMENTED_DIR}")
        shutil.rmtree(AUGMENTED_DIR)

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    for clase in COMMANDS.keys():
        (AUGMENTED_DIR / clase).mkdir(parents=True, exist_ok=True)


def guardar_audio(path, audio, sample_rate):
    audio = normalize_audio(audio)
    sf.write(str(path), audio, sample_rate)


# =========================================================
# PROCESO PRINCIPAL
# =========================================================

def augmentar_dataset():
    preparar_carpeta_salida()

    total_originales = 0
    total_generados = 0

    print("\n=== DATA AUGMENTATION DEL DATASET ===\n")

    for clase in COMMANDS.keys():
        carpeta_origen = SOURCE_DATASET_DIR / clase
        carpeta_destino = AUGMENTED_DIR / clase

        if not carpeta_origen.exists():
            print(f"No existe carpeta: {carpeta_origen}")
            continue

        archivos = sorted(carpeta_origen.glob("*.wav"))

        print(f"\nClase: {clase}")
        print(f"Audios originales encontrados: {len(archivos)}")

        for archivo in archivos:
            audio, sr = sf.read(str(archivo), dtype="float32")

            if sr != SAMPLE_RATE:
                print(f"Advertencia: {archivo.name} tiene sample rate {sr}, esperado {SAMPLE_RATE}")

            # Si el audio viene en estéreo, convertir a mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = normalize_audio(audio)

            # 1. Guardar copia original
            salida_original = carpeta_destino / archivo.name
            guardar_audio(salida_original, audio, sr)
            total_originales += 1

            base_name = archivo.stem

            # 2. Time shifting
            audio_shift = time_shift(audio)
            salida_shift = carpeta_destino / f"{base_name}_aug_shift.wav"
            guardar_audio(salida_shift, audio_shift, sr)
            total_generados += 1

            # 3. Ruido gaussiano
            audio_noise = add_gaussian_noise(audio)
            salida_noise = carpeta_destino / f"{base_name}_aug_noise.wav"
            guardar_audio(salida_noise, audio_noise, sr)
            total_generados += 1

            # 4. Time stretching, versión rápida o lenta aleatoria
            rate = random.choice([0.90, 1.10])
            audio_stretch = time_stretch_simple(audio, rate=rate)
            salida_stretch = carpeta_destino / f"{base_name}_aug_stretch.wav"
            guardar_audio(salida_stretch, audio_stretch, sr)
            total_generados += 1

            # 5. Extra opcional: volumen
            factor = random.choice([0.80, 1.20])
            audio_volume = change_volume(audio, factor=factor)
            salida_volume = carpeta_destino / f"{base_name}_aug_volume.wav"
            guardar_audio(salida_volume, audio_volume, sr)
            total_generados += 1

        total_clase = len(list(carpeta_destino.glob("*.wav")))
        print(f"Total en dataset aumentado para {clase}: {total_clase}")

    print("\n=== RESUMEN ===")
    print(f"Audios originales copiados: {total_originales}")
    print(f"Audios aumentados generados: {total_generados}")
    print(f"Total final aproximado: {total_originales + total_generados}")
    print(f"Dataset aumentado guardado en: {AUGMENTED_DIR}")


if __name__ == "__main__":
    augmentar_dataset()