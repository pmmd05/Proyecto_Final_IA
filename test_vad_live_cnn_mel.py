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
    MEL_BINS,
    LOWER_EDGE_HERTZ,
    UPPER_EDGE_HERTZ,
)


# =========================================================
# MODELO
# =========================================================

MODEL_PATH = MODELS_DIR / "domotica_cnn_mel.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names_cnn_mel.json"

BACKGROUND_CLASS = "RUIDO_FONDO"


# =========================================================
# VAD
# =========================================================

BLOCK_SECONDS = 0.05
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)

PRE_ROLL_SECONDS = 0.35
PRE_ROLL_BLOCKS = int(PRE_ROLL_SECONDS / BLOCK_SECONDS)

BASE_ENERGY_THRESHOLD = 0.020
MANUAL_ENERGY_THRESHOLD = None

USE_ZCR = True
ZCR_MIN = 0.005
ZCR_MAX = 0.35

MIN_SPEECH_BLOCKS = 2

END_SILENCE_SECONDS = 0.50
END_SILENCE_BLOCKS = int(END_SILENCE_SECONDS / BLOCK_SECONDS)

MIN_UTTERANCE_SECONDS = 0.45
MAX_UTTERANCE_SECONDS = 3.0

COOLDOWN_SECONDS = 0.50


ACTION_LABELS = {
    "LUCES_ON": "ENCENDER_LUCES",
    "LUCES_OFF": "APAGAR_LUCES",
    "AIRE_ON": "ENCENDER_AIRE",
    "AIRE_OFF": "APAGAR_AIRE",
    "BOMBA_REGAR": "REGAR",
}


audio_queue = queue.Queue()


# =========================================================
# STREAM
# =========================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio_queue.put(indata.copy())


# =========================================================
# CARGA MODELO
# =========================================================

def cargar_modelo_y_clases():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {CLASS_NAMES_PATH}")

    modelo = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as archivo:
        nombres_clases = json.load(archivo)

    if BACKGROUND_CLASS not in nombres_clases:
        raise ValueError("No se encontró RUIDO_FONDO en class_names_cnn_mel.json")

    return modelo, nombres_clases


# =========================================================
# AUDIO
# =========================================================

def calcular_energia(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio))))


def calcular_zcr(audio):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    if len(audio) < 2:
        return 0.0

    cruces = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
    return float(cruces / len(audio))


def normalizar_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio


def ajustar_duracion_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.squeeze(audio)

    if len(audio) < AUDIO_SAMPLES:
        faltante = AUDIO_SAMPLES - len(audio)
        audio = np.pad(audio, (0, faltante), mode="constant")
    else:
        audio = audio[:AUDIO_SAMPLES]

    return audio


def limpiar_cola_audio():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


# =========================================================
# CALIBRACIÓN
# =========================================================

def calibrar_ruido_ambiente(segundos=2.0):
    if MANUAL_ENERGY_THRESHOLD is not None:
        print("\nUsando umbral manual de energía.")
        print(f"ENERGY_THRESHOLD usado: {MANUAL_ENERGY_THRESHOLD:.5f}\n")
        return MANUAL_ENERGY_THRESHOLD

    print(f"\nCalibrando ruido ambiente durante {segundos:.1f} segundos...")
    print("No hables durante la calibración.\n")

    muestras = []
    inicio = time.time()

    while time.time() - inicio < segundos:
        try:
            bloque = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        bloque = np.squeeze(bloque)
        energia = calcular_energia(bloque)
        muestras.append(energia)

    if len(muestras) == 0:
        return BASE_ENERGY_THRESHOLD

    muestras = np.array(muestras)

    ruido_promedio = float(np.mean(muestras))
    ruido_mediana = float(np.median(muestras))
    ruido_p90 = float(np.percentile(muestras, 90))
    ruido_p95 = float(np.percentile(muestras, 95))

    threshold = max(
        BASE_ENERGY_THRESHOLD,
        ruido_mediana * 1.8,
        ruido_p90 * 1.2
    )

    threshold = min(threshold, 0.25)

    print(f"Energía promedio ruido: {ruido_promedio:.5f}")
    print(f"Energía mediana ruido:  {ruido_mediana:.5f}")
    print(f"Energía p90 ruido:      {ruido_p90:.5f}")
    print(f"Energía p95 ruido:      {ruido_p95:.5f}")
    print(f"ENERGY_THRESHOLD usado: {threshold:.5f}\n")

    return threshold


# =========================================================
# VAD
# =========================================================

def bloque_tiene_voz(audio, energy_threshold):
    energia = calcular_energia(audio)
    zcr = calcular_zcr(audio)

    energia_ok = energia >= energy_threshold

    if USE_ZCR:
        zcr_ok = ZCR_MIN <= zcr <= ZCR_MAX
    else:
        zcr_ok = True

    return energia_ok and zcr_ok, energia, zcr


def obtener_siguiente_frase(energy_threshold):
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

        tiene_voz, energia, zcr = bloque_tiene_voz(
            bloque,
            energy_threshold
        )

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
# LOG MEL PARA CNN
# =========================================================

def convertir_a_log_mel_spectrogram(audio):
    audio = ajustar_duracion_audio(audio)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP
    )

    spectrogram = tf.abs(stft)

    num_spectrogram_bins = FRAME_LENGTH // 2 + 1

    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_EDGE_HERTZ,
        upper_edge_hertz=UPPER_EDGE_HERTZ
    )

    mel_spectrogram = tf.matmul(
        spectrogram,
        mel_weight_matrix
    )

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    media = tf.reduce_mean(log_mel_spectrogram)
    desviacion = tf.math.reduce_std(log_mel_spectrogram)

    log_mel_spectrogram = (
        log_mel_spectrogram - media
    ) / (desviacion + 1e-6)

    # tiempo x mel x canal
    log_mel_spectrogram = log_mel_spectrogram[..., tf.newaxis]

    # batch x tiempo x mel x canal
    entrada = log_mel_spectrogram[tf.newaxis, ...]

    return entrada


# =========================================================
# PREDICCIÓN
# =========================================================

def predecir(modelo, nombres_clases, audio):
    entrada = convertir_a_log_mel_spectrogram(audio)

    predicciones = modelo.predict(entrada, verbose=0)[0]

    indice = int(np.argmax(predicciones))
    clase = nombres_clases[indice]
    confianza = float(predicciones[indice])

    return clase, confianza, predicciones


def mostrar_top3(predicciones, nombres_clases):
    indices = np.argsort(predicciones)[::-1][:3]

    print("Top 3:")

    for posicion, indice in enumerate(indices, start=1):
        clase = nombres_clases[indice]
        prob = predicciones[indice] * 100
        print(f"{posicion}. {clase:15s} - {prob:.2f}%")


def procesar_frase(modelo, nombres_clases, audio):
    clase, confianza, predicciones = predecir(modelo, nombres_clases, audio)

    print("\n" + "=" * 60)
    print("RESULTADO CNN MEL")
    print("=" * 60)
    print(f"Predicción: {clase}")
    print(f"Confianza: {confianza * 100:.2f}%")
    mostrar_top3(predicciones, nombres_clases)

    if clase == BACKGROUND_CLASS:
        print("\nResultado: RUIDO_FONDO. No se ejecuta ninguna acción.")
        print("=" * 60 + "\n")
        return None

    if confianza < CONFIDENCE_THRESHOLD:
        print("\nResultado: baja confianza. Comando rechazado.")
        print("=" * 60 + "\n")
        return None

    accion = ACTION_LABELS.get(clase, clase)
    comando_arduino = ARDUINO_COMMANDS.get(clase, "--")

    print("\nResultado: comando aceptado.")
    print(f"Clase reconocida: {clase}")
    print(f"Acción lógica:    {accion}")
    print(f"Comando Arduino:  {comando_arduino}")
    print("=" * 60 + "\n")

    return clase


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n=== PRUEBA VAD + CNN MEL-SPECTROGRAM ===\n")

    modelo, nombres_clases = cargar_modelo_y_clases()

    print("Modelo CNN Mel cargado correctamente.")
    print("\nClases del modelo:")

    for clase in nombres_clases:
        print(f"- {clase}")

    print("\nFuncionamiento:")
    print("- Micrófono continuo.")
    print("- VAD segmenta la frase.")
    print("- Audio pasa a Log Mel-Spectrogram.")
    print("- CNN 2D clasifica el comando.")
    print("- Presiona CTRL + C para salir.\n")

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