import json
import time

import numpy as np
import sounddevice as sd
import tensorflow as tf

from config import (
    MODELS_DIR,
    SAMPLE_RATE,
    DURATION_SECONDS,
    AUDIO_SAMPLES,
    FRAME_LENGTH,
    FRAME_STEP,
    CONFIDENCE_THRESHOLD,
)


# =========================================================
# CARGA DEL MODELO Y CLASES
# =========================================================

def cargar_modelo_y_clases():
    ruta_modelo = MODELS_DIR / "domotica_gru.keras"
    ruta_clases = MODELS_DIR / "class_names_gru.json"

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    if not ruta_clases.exists():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {ruta_clases}")

    modelo = tf.keras.models.load_model(ruta_modelo)

    with open(ruta_clases, "r", encoding="utf-8") as archivo:
        nombres_clases = json.load(archivo)

    return modelo, nombres_clases


# =========================================================
# GRABACIÓN EN VIVO
# =========================================================

def grabar_audio():
    print("\nPresiona ENTER para iniciar la grabación.")
    input()

    print("Grabando...")
    print("Di el comando ahora.")

    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("Grabación finalizada.")

    audio = np.squeeze(audio)

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


# =========================================================
# PREPROCESAMIENTO
# =========================================================

def convertir_a_espectrograma(audio):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    longitud_audio = tf.shape(audio)[0]

    def rellenar():
        faltante = AUDIO_SAMPLES - longitud_audio
        return tf.pad(audio, paddings=[[0, faltante]])

    def recortar():
        return audio[:AUDIO_SAMPLES]

    audio = tf.cond(
        longitud_audio < AUDIO_SAMPLES,
        rellenar,
        recortar
    )

    espectrograma = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP
    )

    espectrograma = tf.abs(espectrograma)
    espectrograma = tf.math.log(espectrograma + 1e-6)

    # Agregar canal para Conv2D
    espectrograma = espectrograma[..., tf.newaxis]

    # Agregar dimensión de batch
    espectrograma = espectrograma[tf.newaxis, ...]

    return espectrograma


# =========================================================
# PREDICCIÓN
# =========================================================

def predecir_comando(modelo, nombres_clases, audio):
    espectrograma = convertir_a_espectrograma(audio)

    predicciones = modelo.predict(espectrograma, verbose=0)[0]

    indice_predicho = int(np.argmax(predicciones))
    clase_predicha = nombres_clases[indice_predicho]
    confianza = float(predicciones[indice_predicho])

    return clase_predicha, confianza, predicciones


def mostrar_resultados(clase_predicha, confianza, predicciones, nombres_clases):
    print("\n" + "=" * 60)
    print("RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)

    print(f"Predicción principal: {clase_predicha}")
    print(f"Confianza: {confianza * 100:.2f}%")

    if confianza >= CONFIDENCE_THRESHOLD:
        print("Estado: COMANDO ACEPTADO")
    else:
        print("Estado: RECHAZADO POR BAJA CONFIANZA")

    print("\nTop 3 predicciones:")

    indices_top = np.argsort(predicciones)[::-1][:3]

    for posicion, indice in enumerate(indices_top, start=1):
        clase = nombres_clases[indice]
        probabilidad = predicciones[indice] * 100
        print(f"{posicion}. {clase:20s} - {probabilidad:.2f}%")

    print("=" * 60)


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    print("\n=== PRUEBA EN VIVO DEL MODELO GRU ===")

    modelo, nombres_clases = cargar_modelo_y_clases()

    print("\nModelo cargado correctamente.")
    print("Clases disponibles:")

    for clase in nombres_clases:
        print(f"- {clase}")

    print("\nInstrucciones:")
    print("- Presiona ENTER cuando estés lista para grabar.")
    print("- Di uno de los comandos.")
    print("- El programa mostrará la clase reconocida.")
    print("- Escribe CTRL + C para salir.")

    try:
        while True:
            audio = grabar_audio()

            clase_predicha, confianza, predicciones = predecir_comando(
                modelo,
                nombres_clases,
                audio
            )

            mostrar_resultados(
                clase_predicha,
                confianza,
                predicciones,
                nombres_clases
            )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nPrueba finalizada.")


if __name__ == "__main__":
    main()