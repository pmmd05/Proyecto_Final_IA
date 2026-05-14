import json
import time
import queue
from collections import deque

import numpy as np
import sounddevice as sd
import tensorflow as tf

from config import (
    MODELS_DIR,
    SAMPLE_RATE,
    AUDIO_SAMPLES,
    FRAME_LENGTH,
    FRAME_STEP,
    CONFIDENCE_THRESHOLD,
    ARDUINO_COMMANDS,
)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

BACKGROUND_CLASS = "RUIDO_FONDO"

MODEL_PATH = MODELS_DIR / "domotica_gru.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names_gru.json"

# El micrófono está activo continuamente.
# Internamente se procesa en bloques pequeños.
BLOCK_SECONDS = 0.05
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)

# Guarda audio previo para no cortar el inicio de la palabra.
PRE_ROLL_SECONDS = 0.35
PRE_ROLL_BLOCKS = int(PRE_ROLL_SECONDS / BLOCK_SECONDS)

# VAD por energía.
# Si detecta demasiado ruido, sube el umbral manual.
# Si no detecta tu voz, bájalo.
BASE_ENERGY_THRESHOLD = 0.020
MANUAL_ENERGY_THRESHOLD = None

# VAD por cruces por cero.
USE_ZCR = True
ZCR_MIN = 0.005
ZCR_MAX = 0.35

# Bloques consecutivos necesarios para confirmar inicio de voz.
MIN_SPEECH_BLOCKS = 2

# Silencio necesario para confirmar fin de frase.
END_SILENCE_SECONDS = 0.50
END_SILENCE_BLOCKS = int(END_SILENCE_SECONDS / BLOCK_SECONDS)

# Duraciones aceptables.
MIN_UTTERANCE_SECONDS = 0.45
MAX_UTTERANCE_SECONDS = 3.00

# Pausa después de clasificar para evitar doble detección.
COOLDOWN_SECONDS = 0.50

# Umbral de confianza del modelo.
COMMAND_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD


# =========================================================
# ETIQUETAS PARA MOSTRAR EN PANTALLA
# =========================================================

ACTION_LABELS = {
    "LUCES_ON": "ENCENDER_LUCES",
    "LUCES_OFF": "APAGAR_LUCES",
    "AIRE_ON": "ENCENDER_AIRE",
    "AIRE_OFF": "APAGAR_AIRE",
    "BOMBA_REGAR": "REGAR",
}


# =========================================================
# COLA GLOBAL DE AUDIO
# =========================================================

audio_queue = queue.Queue()


# =========================================================
# CALLBACK DEL MICRÓFONO CONTINUO
# =========================================================

def audio_callback(indata, frames, time_info, status):
    """
    Esta función recibe audio del micrófono de forma continua.
    El usuario no tiene que presionar nada.
    """
    if status:
        print(status)

    audio_queue.put(indata.copy())


# =========================================================
# CARGA DEL MODELO GRU
# =========================================================

def cargar_modelo_y_clases():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo GRU: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {CLASS_NAMES_PATH}")

    modelo = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as archivo:
        nombres_clases = json.load(archivo)

    if BACKGROUND_CLASS not in nombres_clases:
        raise ValueError(
            f"No se encontró la clase {BACKGROUND_CLASS}. "
            "Verifica que el modelo GRU fue entrenado con RUIDO_FONDO."
        )

    return modelo, nombres_clases


# =========================================================
# FUNCIONES DE AUDIO
# =========================================================

def calcular_energia(audio):
    """
    Energía RMS del bloque.
    Se usa para detectar si hay sonido suficientemente fuerte.
    """
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio))))


def calcular_zcr(audio):
    """
    Zero Crossing Rate.
    Mide cuántas veces la señal cruza por cero.
    Ayuda a distinguir voz de silencios o ruidos muy planos.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    if len(audio) < 2:
        return 0.0

    cruces = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
    zcr = cruces / len(audio)

    return float(zcr)


def normalizar_audio(audio):
    """
    Normaliza la amplitud del audio.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


def ajustar_duracion_audio(audio):
    """
    Ajusta el audio al tamaño usado durante el entrenamiento.
    Si es corto, rellena con ceros.
    Si es largo, recorta.
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    if len(audio) < AUDIO_SAMPLES:
        faltante = AUDIO_SAMPLES - len(audio)
        audio = np.pad(audio, (0, faltante), mode="constant")
    else:
        audio = audio[:AUDIO_SAMPLES]

    return audio


def limpiar_cola_audio():
    """
    Limpia audio atrasado de la cola.
    """
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# =========================================================
# CALIBRACIÓN DEL RUIDO AMBIENTE
# =========================================================

def calibrar_ruido_ambiente(segundos=2.0):
    """
    Calcula un umbral de energía basado en el ambiente.
    Usa percentiles para evitar que un pico aislado suba demasiado el umbral.
    """

    if MANUAL_ENERGY_THRESHOLD is not None:
        print("\nUsando umbral manual de energía.")
        print(f"ENERGY_THRESHOLD usado: {MANUAL_ENERGY_THRESHOLD:.5f}\n")
        return MANUAL_ENERGY_THRESHOLD

    print(f"\nCalibrando ruido ambiente durante {segundos:.1f} segundos...")
    print("No hables durante la calibración.\n")

    muestras_energia = []

    inicio = time.time()

    while time.time() - inicio < segundos:
        try:
            bloque = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        bloque = np.squeeze(bloque)
        energia = calcular_energia(bloque)
        muestras_energia.append(energia)

    if len(muestras_energia) == 0:
        print("No se pudo calibrar. Usando umbral base.")
        return BASE_ENERGY_THRESHOLD

    muestras_energia = np.array(muestras_energia)

    ruido_promedio = float(np.mean(muestras_energia))
    ruido_mediana = float(np.median(muestras_energia))
    ruido_p90 = float(np.percentile(muestras_energia, 90))
    ruido_p95 = float(np.percentile(muestras_energia, 95))

    threshold = max(
        BASE_ENERGY_THRESHOLD,
        ruido_mediana * 1.8,
        ruido_p90 * 1.2
    )

    # Límite superior para evitar que un pico haga imposible detectar voz.
    threshold = min(threshold, 0.25)

    print(f"Energía promedio ruido: {ruido_promedio:.5f}")
    print(f"Energía mediana ruido:  {ruido_mediana:.5f}")
    print(f"Energía p90 ruido:      {ruido_p90:.5f}")
    print(f"Energía p95 ruido:      {ruido_p95:.5f}")
    print(f"ENERGY_THRESHOLD usado: {threshold:.5f}\n")

    return threshold


# =========================================================
# PREPROCESAMIENTO PARA GRU
# =========================================================

def convertir_a_secuencia_espectral(audio):
    """
    Convierte el audio segmentado a una secuencia espectral compatible con la GRU.

    La GRU espera:
    pasos_de_tiempo x características_de_frecuencia

    Este preprocesamiento debe coincidir con train_model_gru.py.
    """
    audio = ajustar_duracion_audio(audio)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    espectrograma = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP
    )

    espectrograma = tf.abs(espectrograma)
    espectrograma = tf.math.log(espectrograma + 1e-6)

    media = tf.reduce_mean(espectrograma)
    desviacion = tf.math.reduce_std(espectrograma)

    espectrograma = (espectrograma - media) / (desviacion + 1e-6)

    # Batch dimension.
    secuencia = espectrograma[tf.newaxis, ...]

    return secuencia


def predecir(modelo, nombres_clases, audio):
    secuencia = convertir_a_secuencia_espectral(audio)

    predicciones = modelo.predict(secuencia, verbose=0)[0]

    indice = int(np.argmax(predicciones))
    clase = nombres_clases[indice]
    confianza = float(predicciones[indice])

    return clase, confianza, predicciones


def mostrar_top3(predicciones, nombres_clases):
    indices = np.argsort(predicciones)[::-1][:3]

    print("Top 3:")
    for posicion, indice in enumerate(indices, start=1):
        clase = nombres_clases[indice]
        probabilidad = predicciones[indice] * 100
        print(f"{posicion}. {clase:15s} - {probabilidad:.2f}%")


# =========================================================
# VAD: DETECCIÓN Y SEGMENTACIÓN AUTOMÁTICA
# =========================================================

def bloque_tiene_voz(audio, energy_threshold):
    """
    Decide si un bloque contiene voz usando:
    - Energía RMS
    - Cruces por cero
    """
    energia = calcular_energia(audio)
    zcr = calcular_zcr(audio)

    energia_ok = energia >= energy_threshold

    if USE_ZCR:
        zcr_ok = ZCR_MIN <= zcr <= ZCR_MAX
    else:
        zcr_ok = True

    return energia_ok and zcr_ok, energia, zcr


def obtener_siguiente_frase(energy_threshold):
    """
    Segmenta automáticamente una frase de voz.

    Funcionamiento:
    1. Escucha continuamente.
    2. Usa pre-buffer para no cortar el inicio.
    3. Detecta inicio de voz por energía/ZCR.
    4. Acumula bloques mientras hay voz.
    5. Detecta fin por silencio.
    6. Devuelve solo el tramo segmentado y normalizado.
    """

    pre_buffer = deque(maxlen=PRE_ROLL_BLOCKS)

    speech_blocks = []
    speech_started = False

    consecutive_speech = 0
    consecutive_silence = 0

    while True:
        try:
            bloque = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        bloque = np.squeeze(bloque)
        tiene_voz, energia, zcr = bloque_tiene_voz(bloque, energy_threshold)

        if not speech_started:
            pre_buffer.append(bloque)

            print(
                f"Escuchando... Energía: {energia:.5f} | ZCR: {zcr:.5f}",
                end="\r"
            )

            if tiene_voz:
                consecutive_speech += 1
            else:
                consecutive_speech = 0

            if consecutive_speech >= MIN_SPEECH_BLOCKS:
                speech_started = True
                speech_blocks = list(pre_buffer)
                consecutive_silence = 0
                print("\nInicio de voz detectado.")

        else:
            speech_blocks.append(bloque)

            if tiene_voz:
                consecutive_silence = 0
            else:
                consecutive_silence += 1

            duracion_actual = len(speech_blocks) * BLOCK_SECONDS

            if consecutive_silence >= END_SILENCE_BLOCKS:
                audio = np.concatenate(speech_blocks)
                audio = normalizar_audio(audio)

                duracion = len(audio) / SAMPLE_RATE

                print(f"Fin de frase detectado. Duración: {duracion:.2f} s")

                if duracion < MIN_UTTERANCE_SECONDS:
                    print("Frase demasiado corta. Se ignora.\n")
                    return None

                return audio

            if duracion_actual >= MAX_UTTERANCE_SECONDS:
                audio = np.concatenate(speech_blocks)
                audio = normalizar_audio(audio)

                duracion = len(audio) / SAMPLE_RATE

                print(f"Frase larga cortada automáticamente. Duración: {duracion:.2f} s")

                return audio


# =========================================================
# CLASIFICACIÓN DE COMANDOS
# =========================================================

def procesar_frase(modelo, nombres_clases, audio):
    clase, confianza, predicciones = predecir(
        modelo,
        nombres_clases,
        audio
    )

    print("\n" + "=" * 60)
    print("RESULTADO DE CLASIFICACIÓN - GRU")
    print("=" * 60)
    print(f"Predicción: {clase}")
    print(f"Confianza: {confianza * 100:.2f}%")
    mostrar_top3(predicciones, nombres_clases)

    if clase == BACKGROUND_CLASS:
        print("\nResultado: RUIDO_FONDO. No se ejecuta ninguna acción.")
        print("=" * 60 + "\n")
        return None

    if confianza < COMMAND_CONFIDENCE_THRESHOLD:
        print("\nResultado: baja confianza. Comando rechazado.")
        print("=" * 60 + "\n")
        return None

    accion = ACTION_LABELS.get(clase, clase)
    comando_arduino = ARDUINO_COMMANDS.get(clase, "SIN_COMANDO_ARDUINO")

    print("\nResultado: comando aceptado.")
    print(f"Clase reconocida: {clase}")
    print(f"Acción lógica:    {accion}")
    print(f"Comando Arduino:  {comando_arduino}")
    print("=" * 60 + "\n")

    return clase


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    print("\n=== VAD CONTINUO + MODELO SECUENCIAL GRU ===\n")

    modelo, nombres_clases = cargar_modelo_y_clases()

    print("Modelo GRU cargado correctamente.")
    print("\nClases del modelo:")
    for clase in nombres_clases:
        print(f"- {clase}")

    print("\nFuncionamiento:")
    print("1. El micrófono permanece escuchando continuamente.")
    print("2. El VAD segmenta automáticamente el tramo con voz.")
    print("3. Se descartan silencios iniciales y finales.")
    print("4. Se normaliza la amplitud.")
    print("5. Se extrae la secuencia espectral.")
    print("6. La GRU clasifica el comando.")
    print("7. Si es RUIDO_FONDO o baja confianza, se ignora.\n")

    print("Presiona CTRL + C para salir.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    ):
        limpiar_cola_audio()
        energy_threshold = calibrar_ruido_ambiente(segundos=2.0)
        limpiar_cola_audio()

        try:
            while True:
                audio_frase = obtener_siguiente_frase(energy_threshold)

                if audio_frase is None:
                    continue

                procesar_frase(
                    modelo,
                    nombres_clases,
                    audio_frase
                )

                time.sleep(COOLDOWN_SECONDS)

        except KeyboardInterrupt:
            print("\nPrueba finalizada.")


if __name__ == "__main__":
    main()