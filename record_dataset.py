import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import DATASET_DIR, COMMANDS, SAMPLE_RATE, DURATION_SECONDS


def crear_carpetas():
    """
    Crea la carpeta principal del dataset y una subcarpeta por cada clase.
    """
    DATASET_DIR.mkdir(exist_ok=True)

    for etiqueta in COMMANDS:
        carpeta = DATASET_DIR / etiqueta
        carpeta.mkdir(parents=True, exist_ok=True)


def grabar_audio(duracion, sample_rate):
    """
    Graba audio desde el micrófono de la computadora.
    """
    print("Grabando...")

    audio = sd.rec(
        int(duracion * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return np.squeeze(audio)


def guardar_audio(ruta_salida, audio, sample_rate):
    """
    Guarda el audio en formato .wav.
    También normaliza ligeramente el volumen para evitar audios demasiado bajos.
    """
    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

    if max_val > 0:
        audio = audio / max_val * 0.9

    sf.write(str(ruta_salida), audio, sample_rate)


def pedir_datos_grabacion():
    """
    Solicita los datos básicos de la persona y el entorno.
    """
    print("\n=== Recolector de audios para Domótica por Voz ===\n")

    print("Clases disponibles:")
    for etiqueta, frase in COMMANDS.items():
        print(f"- {etiqueta}: {frase}")

    persona_id = input("\nID de persona, ejemplo p01: ").strip()
    entorno_id = input("ID de entorno, ejemplo e01 o e02: ").strip()
    repeticiones = int(input("Repeticiones por clase en este entorno: ").strip())

    return persona_id, entorno_id, repeticiones


def main():
    crear_carpetas()

    persona_id, entorno_id, repeticiones = pedir_datos_grabacion()

    print("\nRecomendaciones antes de grabar:")
    print("- e01 puede ser un entorno silencioso.")
    print("- e02 puede ser un entorno con ruido moderado.")
    print("- Habla normal, sin exagerar la voz.")
    print("- Deja una pequeña pausa antes y después de cada frase.")
    print("- Para RUIDO_FONDO puedes guardar silencio, ruido ambiente o habla no relacionada.")

    input("\nPresiona ENTER para iniciar...")

    for etiqueta, frase in COMMANDS.items():
        print("\n" + "=" * 60)
        print(f"Clase actual: {etiqueta}")
        print(f"Contenido a grabar: {frase}")
        print("=" * 60)

        for repeticion in range(1, repeticiones + 1):
            nombre_archivo = f"{persona_id}_{entorno_id}_r{repeticion:02d}.wav"
            ruta_salida = DATASET_DIR / etiqueta / nombre_archivo

            if ruta_salida.exists():
                print(f"Ya existe, se omite: {ruta_salida}")
                continue

            print(f"\nGrabación {repeticion}/{repeticiones}")
            print(f"Di: {frase}")

            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)

            audio = grabar_audio(DURATION_SECONDS, SAMPLE_RATE)
            guardar_audio(ruta_salida, audio, SAMPLE_RATE)

            print(f"Guardado: {ruta_salida}")

            time.sleep(0.5)

    print("\nProceso terminado.")
    print("Puedes revisar las carpetas dentro de dataset_domotica_voz.")


if __name__ == "__main__":
    main()