import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from config import (
    DATASET_DIR,
    MODELS_DIR,
    COMMANDS,
    AUDIO_SAMPLES,
    FRAME_LENGTH,
    FRAME_STEP,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)


# =========================================================
# CONFIGURACIÓN DEL SEGUNDO ENTRENAMIENTO
# =========================================================

CONTINUE_EPOCHS = 15
FINE_TUNE_LEARNING_RATE = 0.0001

MODEL_INPUT = MODELS_DIR / "domotica_cnn.keras"
CLASS_NAMES_INPUT = MODELS_DIR / "class_names.json"

MODEL_OUTPUT = MODELS_DIR / "domotica_cnn_v2.keras"


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# PREPROCESAMIENTO
# =========================================================

def cargar_audio(ruta_archivo):
    audio_binario = tf.io.read_file(ruta_archivo)

    audio, sample_rate = tf.audio.decode_wav(
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


def convertir_a_espectrograma(audio):
    espectrograma = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP
    )

    espectrograma = tf.abs(espectrograma)
    espectrograma = tf.math.log(espectrograma + 1e-6)
    espectrograma = espectrograma[..., tf.newaxis]

    return espectrograma


def preprocesar_audio(ruta_archivo, etiqueta):
    audio = cargar_audio(ruta_archivo)
    espectrograma = convertir_a_espectrograma(audio)
    return espectrograma, etiqueta


# =========================================================
# DATASET
# =========================================================

def cargar_clases_guardadas():
    if not CLASS_NAMES_INPUT.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de clases: {CLASS_NAMES_INPUT}"
        )

    with open(CLASS_NAMES_INPUT, "r", encoding="utf-8") as archivo:
        nombres_clases = json.load(archivo)

    return nombres_clases


def construir_lista_archivos(nombres_clases):
    archivos = []
    etiquetas = []

    for indice, clase in enumerate(nombres_clases):
        carpeta = DATASET_DIR / clase

        if not carpeta.exists():
            print(f"Advertencia: no existe la carpeta {carpeta}")
            continue

        archivos_wav = sorted(carpeta.glob("*.wav"))

        for archivo in archivos_wav:
            archivos.append(str(archivo))
            etiquetas.append(indice)

    archivos = np.array(archivos)
    etiquetas = np.array(etiquetas)

    if len(archivos) == 0:
        raise RuntimeError("No se encontraron archivos .wav en el dataset.")

    return archivos, etiquetas


def dividir_dataset(archivos, etiquetas):
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

    return (
        archivos_train,
        etiquetas_train,
        archivos_val,
        etiquetas_val,
        archivos_test,
        etiquetas_test
    )


def crear_tf_dataset(archivos, etiquetas, mezclar=False):
    dataset = tf.data.Dataset.from_tensor_slices((archivos, etiquetas))

    if mezclar:
        dataset = dataset.shuffle(
            buffer_size=len(archivos),
            seed=RANDOM_SEED
        )

    dataset = dataset.map(
        preprocesar_audio,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =========================================================
# EVALUACIÓN
# =========================================================

def guardar_grafica_entrenamiento(historial, ruta_salida):
    plt.figure(figsize=(8, 5))

    plt.plot(historial.history["accuracy"], label="Entrenamiento")
    plt.plot(historial.history["val_accuracy"], label="Validación")

    plt.title("Exactitud durante entrenamiento continuo")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(ruta_salida, dpi=160, bbox_inches="tight")
    plt.close()


def guardar_grafica_perdida(historial, ruta_salida):
    plt.figure(figsize=(8, 5))

    plt.plot(historial.history["loss"], label="Entrenamiento")
    plt.plot(historial.history["val_loss"], label="Validación")

    plt.title("Pérdida durante entrenamiento continuo")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(ruta_salida, dpi=160, bbox_inches="tight")
    plt.close()


def evaluar_modelo(modelo, dataset_prueba, nombres_clases):
    y_real = []
    y_predicho = []

    for x_batch, y_batch in dataset_prueba:
        predicciones = modelo.predict(x_batch, verbose=0)
        clases_predichas = np.argmax(predicciones, axis=1)

        y_real.extend(y_batch.numpy())
        y_predicho.extend(clases_predichas)

    print("\n=== REPORTE DE CLASIFICACIÓN - MODELO V2 ===\n")

    print(
        classification_report(
            y_real,
            y_predicho,
            target_names=nombres_clases,
            zero_division=0
        )
    )

    matriz = confusion_matrix(
        y_real,
        y_predicho,
        labels=list(range(len(nombres_clases)))
    )

    figura, eje = plt.subplots(figsize=(12, 12))

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

    plt.title("Matriz de confusión - Modelo V2")
    plt.savefig(
        MODELS_DIR / "confusion_matrix_v2.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n=== ENTRENAMIENTO CONTINUO CNN - DOMÓTICA POR VOZ ===\n")

    if not MODEL_INPUT.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo base: {MODEL_INPUT}"
        )

    nombres_clases = cargar_clases_guardadas()

    print("Clases cargadas desde class_names.json:")
    for i, clase in enumerate(nombres_clases):
        print(f"{i:02d}. {clase}")

    archivos, etiquetas = construir_lista_archivos(nombres_clases)

    print(f"\nTotal de audios encontrados actualmente: {len(archivos)}")

    print("\nConteo por clase:")
    for i, clase in enumerate(nombres_clases):
        cantidad = np.sum(etiquetas == i)
        print(f"{clase:20s}: {cantidad} audios")

    (
        archivos_train,
        etiquetas_train,
        archivos_val,
        etiquetas_val,
        archivos_test,
        etiquetas_test
    ) = dividir_dataset(archivos, etiquetas)

    print("\nDivisión del dataset actualizado:")
    print(f"Entrenamiento: {len(archivos_train)} audios")
    print(f"Validación:    {len(archivos_val)} audios")
    print(f"Prueba:        {len(archivos_test)} audios")

    dataset_train = crear_tf_dataset(
        archivos_train,
        etiquetas_train,
        mezclar=True
    )

    dataset_val = crear_tf_dataset(
        archivos_val,
        etiquetas_val,
        mezclar=False
    )

    dataset_test = crear_tf_dataset(
        archivos_test,
        etiquetas_test,
        mezclar=False
    )

    print("\nCargando modelo anterior...")
    modelo = tf.keras.models.load_model(MODEL_INPUT)

    print("\nRecompilando modelo con tasa de aprendizaje más baja...")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nIniciando entrenamiento continuo...\n")

    historial = modelo.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=CONTINUE_EPOCHS,
        callbacks=callbacks
    )

    print("\nGuardando modelo actualizado...")

    modelo.save(MODEL_OUTPUT)

    guardar_grafica_entrenamiento(
        historial,
        MODELS_DIR / "training_accuracy_v2.png"
    )

    guardar_grafica_perdida(
        historial,
        MODELS_DIR / "training_loss_v2.png"
    )

    evaluar_modelo(
        modelo,
        dataset_test,
        nombres_clases
    )

    print("\nEntrenamiento continuo terminado.")
    print("\nArchivos generados:")
    print(f"- {MODEL_OUTPUT}")
    print(f"- {MODELS_DIR / 'training_accuracy_v2.png'}")
    print(f"- {MODELS_DIR / 'training_loss_v2.png'}")
    print(f"- {MODELS_DIR / 'confusion_matrix_v2.png'}")

    print("\nImportante:")
    print("El modelo original sigue guardado como domotica_cnn.keras.")
    print("El modelo mejorado se guardó como domotica_cnn_v2.keras.")


if __name__ == "__main__":
    main()