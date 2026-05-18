import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

from config import (
    DATASET_DIR,
    MODELS_DIR,
    COMMANDS,
    SAMPLE_RATE,
    AUDIO_SAMPLES,
    FRAME_LENGTH,
    FRAME_STEP,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    MEL_BINS,
    LOWER_EDGE_HERTZ,
    UPPER_EDGE_HERTZ,
)


# =========================================================
# MODELO FINAL A EVALUAR
# =========================================================

MODEL_PATH = MODELS_DIR / "domotica_gru_mel.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names_gru_mel.json"


# =========================================================
# CARGA Y PREPROCESAMIENTO DE AUDIO
# =========================================================

def cargar_audio(ruta_archivo):
    """
    Lee un archivo WAV, lo convierte a mono y lo ajusta
    a la duración fija usada durante el entrenamiento.
    """

    audio_binario = tf.io.read_file(ruta_archivo)

    audio, _ = tf.audio.decode_wav(
        audio_binario,
        desired_channels=1
    )

    audio = tf.squeeze(audio, axis=-1)

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

    return audio


def convertir_a_log_mel_spectrogram(audio):
    """
    Convierte el audio a Log Mel-Spectrogram.

    Salida:
    pasos de tiempo × bandas Mel

    Esta función debe coincidir con el preprocesamiento
    usado durante el entrenamiento del modelo GRU + Mel.
    """

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

    return log_mel_spectrogram


def preprocesar_audio(ruta_archivo, etiqueta):
    audio = cargar_audio(ruta_archivo)
    log_mel = convertir_a_log_mel_spectrogram(audio)

    return log_mel, etiqueta


# =========================================================
# CONSTRUCCIÓN DEL DATASET
# =========================================================

def construir_lista_archivos():
    """
    Lee las carpetas del dataset y construye:
    - lista de archivos
    - etiquetas numéricas
    - nombres de clases
    """

    nombres_clases = list(COMMANDS.keys())

    archivos = []
    etiquetas = []

    for indice, clase in enumerate(nombres_clases):
        carpeta_clase = DATASET_DIR / clase

        if not carpeta_clase.exists():
            print(f"Advertencia: no existe la carpeta {carpeta_clase}")
            continue

        archivos_wav = sorted(carpeta_clase.glob("*.wav"))

        for archivo in archivos_wav:
            archivos.append(str(archivo))
            etiquetas.append(indice)

    archivos = np.array(archivos)
    etiquetas = np.array(etiquetas)

    if len(archivos) == 0:
        raise RuntimeError("No se encontraron archivos .wav en el dataset.")

    return archivos, etiquetas, nombres_clases


def dividir_dataset(archivos, etiquetas):
    """
    Reconstruye la misma división train / validation / test
    usada durante el entrenamiento.
    """

    test_size_total = TEST_SPLIT + VALIDATION_SPLIT

    archivos_train, archivos_temp, etiquetas_train, etiquetas_temp = train_test_split(
        archivos,
        etiquetas,
        test_size=test_size_total,
        random_state=RANDOM_SEED,
        stratify=etiquetas
    )

    proporcion_validacion = VALIDATION_SPLIT / test_size_total

    archivos_val, archivos_test, etiquetas_val, etiquetas_test = train_test_split(
        archivos_temp,
        etiquetas_temp,
        test_size=1 - proporcion_validacion,
        random_state=RANDOM_SEED,
        stratify=etiquetas_temp
    )

    return archivos_test, etiquetas_test


def crear_tf_dataset(archivos, etiquetas):
    dataset = tf.data.Dataset.from_tensor_slices((archivos, etiquetas))

    dataset = dataset.map(
        preprocesar_audio,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =========================================================
# EVALUACIÓN
# =========================================================

def guardar_reporte_txt(ruta_salida, accuracy, reporte_texto):
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        archivo.write("=== REPORTE DE CLASIFICACIÓN - GRU + LOG MEL-SPECTROGRAM ===\n\n")
        archivo.write(f"Accuracy general: {accuracy:.4f}\n\n")
        archivo.write(reporte_texto)


def guardar_reporte_csv(ruta_salida, reporte_dict):
    with open(ruta_salida, "w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.writer(archivo_csv)

        writer.writerow([
            "clase",
            "precision",
            "recall",
            "f1_score",
            "support"
        ])

        for clase, metricas in reporte_dict.items():
            if isinstance(metricas, dict):
                writer.writerow([
                    clase,
                    metricas.get("precision", ""),
                    metricas.get("recall", ""),
                    metricas.get("f1-score", ""),
                    metricas.get("support", "")
                ])


def guardar_matriz_confusion_png(matriz, nombres_clases, ruta_salida):
    figura, eje = plt.subplots(figsize=(10, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nombres_clases
    )

    display.plot(
        ax=eje,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Matriz de confusión - GRU + Log Mel-Spectrogram")
    plt.savefig(
        ruta_salida,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()


def main():
    print("\n=== EVALUACIÓN DEL MODELO GRU + LOG MEL-SPECTROGRAM ===\n")

    print(f"Dataset usado: {DATASET_DIR}")
    print(f"Modelo usado:  {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {CLASS_NAMES_PATH}")

    modelo = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as archivo:
        nombres_clases_modelo = json.load(archivo)

    archivos, etiquetas, nombres_clases_dataset = construir_lista_archivos()

    print(f"\nTotal de audios encontrados: {len(archivos)}")

    print("\nClases del dataset:")
    for i, clase in enumerate(nombres_clases_dataset):
        cantidad = np.sum(etiquetas == i)
        print(f"{i:02d}. {clase:20s} -> {cantidad} audios")

    if nombres_clases_modelo != nombres_clases_dataset:
        print("\nAdvertencia:")
        print("Las clases guardadas en el modelo no coinciden exactamente con COMMANDS en config.py.")
        print("Clases del modelo:")
        print(nombres_clases_modelo)
        print("Clases del dataset:")
        print(nombres_clases_dataset)

    archivos_test, etiquetas_test = dividir_dataset(archivos, etiquetas)

    print("\nConjunto de prueba:")
    print(f"Audios de prueba: {len(archivos_test)}")

    dataset_test = crear_tf_dataset(
        archivos_test,
        etiquetas_test
    )

    y_real = []
    y_predicho = []

    print("\nEvaluando modelo...\n")

    for x_batch, y_batch in dataset_test:
        predicciones = modelo.predict(x_batch, verbose=0)
        clases_predichas = np.argmax(predicciones, axis=1)

        y_real.extend(y_batch.numpy())
        y_predicho.extend(clases_predichas)

    y_real = np.array(y_real)
    y_predicho = np.array(y_predicho)

    accuracy = accuracy_score(y_real, y_predicho)

    reporte_texto = classification_report(
        y_real,
        y_predicho,
        target_names=nombres_clases_dataset,
        zero_division=0
    )

    reporte_dict = classification_report(
        y_real,
        y_predicho,
        target_names=nombres_clases_dataset,
        zero_division=0,
        output_dict=True
    )

    matriz = confusion_matrix(
        y_real,
        y_predicho,
        labels=list(range(len(nombres_clases_dataset)))
    )

    print("=== RESULTADOS ===\n")
    print(f"Accuracy general: {accuracy:.4f}\n")
    print("=== REPORTE DE CLASIFICACIÓN ===\n")
    print(reporte_texto)

    # Crear carpeta models si no existe
    MODELS_DIR.mkdir(exist_ok=True)

    ruta_reporte_txt = MODELS_DIR / "classification_report_gru_mel.txt"
    ruta_reporte_csv = MODELS_DIR / "classification_report_gru_mel.csv"
    ruta_matriz_csv = MODELS_DIR / "confusion_matrix_gru_mel.csv"
    ruta_matriz_png = MODELS_DIR / "confusion_matrix_gru_mel_eval.png"

    guardar_reporte_txt(
        ruta_reporte_txt,
        accuracy,
        reporte_texto
    )

    guardar_reporte_csv(
        ruta_reporte_csv,
        reporte_dict
    )

    np.savetxt(
        ruta_matriz_csv,
        matriz,
        delimiter=",",
        fmt="%d"
    )

    guardar_matriz_confusion_png(
        matriz,
        nombres_clases_dataset,
        ruta_matriz_png
    )

    print("\nArchivos generados:")
    print(f"- {ruta_reporte_txt}")
    print(f"- {ruta_reporte_csv}")
    print(f"- {ruta_matriz_csv}")
    print(f"- {ruta_matriz_png}")


if __name__ == "__main__":
    main()