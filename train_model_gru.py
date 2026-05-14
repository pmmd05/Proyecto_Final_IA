import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from config import (
    DATASET_DIR,
    MODELS_DIR,
    COMMANDS,
    AUDIO_SAMPLES,
    FRAME_LENGTH,
    FRAME_STEP,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)


# =========================================================
# SEMILLAS PARA REPRODUCIBILIDAD
# =========================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# CARGA Y PREPROCESAMIENTO DE AUDIO
# =========================================================

def cargar_audio(ruta_archivo):
    """
    Lee un archivo .wav y lo ajusta a una duración fija.
    Si es más corto, rellena con ceros.
    Si es más largo, recorta.
    """
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


def convertir_a_secuencia_espectral(audio):
    """
    Convierte el audio a una representación secuencial para GRU.

    Resultado:
    - Eje 0: pasos de tiempo
    - Eje 1: frecuencias

    Es decir:
    tiempo x características
    """
    espectrograma = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP
    )

    espectrograma = tf.abs(espectrograma)
    espectrograma = tf.math.log(espectrograma + 1e-6)

    # Normalización por muestra para estabilizar el entrenamiento
    media = tf.reduce_mean(espectrograma)
    desviacion = tf.math.reduce_std(espectrograma)

    espectrograma = (espectrograma - media) / (desviacion + 1e-6)

    return espectrograma


def preprocesar_audio(ruta_archivo, etiqueta):
    audio = cargar_audio(ruta_archivo)
    secuencia = convertir_a_secuencia_espectral(audio)
    return secuencia, etiqueta


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
    Divide el dataset en entrenamiento, validación y prueba.
    Usa stratify para mantener proporciones similares por clase.
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
# MODELO GRU
# =========================================================

def crear_modelo_gru(input_shape, numero_clases):
    """
    Modelo secuencial GRU entrenado desde cero.

    La entrada tiene forma:
    pasos_de_tiempo x características_de_frecuencia

    No usa modelos preentrenados.
    """
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                96,
                return_sequences=True,
                dropout=0.25,
                recurrent_dropout=0.0
            )
        ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                64,
                return_sequences=False,
                dropout=0.25,
                recurrent_dropout=0.0
            )
        ),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(numero_clases, activation="softmax")
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


# =========================================================
# GRÁFICAS Y EVALUACIÓN
# =========================================================

def guardar_grafica_entrenamiento(historial, ruta_salida):
    plt.figure(figsize=(8, 5))

    plt.plot(historial.history["accuracy"], label="Entrenamiento")
    plt.plot(historial.history["val_accuracy"], label="Validación")

    plt.title("Exactitud durante el entrenamiento - GRU")
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

    plt.title("Pérdida durante el entrenamiento - GRU")
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

    print("\n=== REPORTE DE CLASIFICACIÓN - GRU ===\n")

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

    plt.title("Matriz de confusión - GRU")
    plt.savefig(
        MODELS_DIR / "confusion_matrix_gru.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n=== ENTRENAMIENTO GRU - DOMÓTICA POR VOZ ===\n")

    print(f"Dataset usado: {DATASET_DIR}")

    archivos, etiquetas, nombres_clases = construir_lista_archivos()

    print(f"\nTotal de audios encontrados: {len(archivos)}")

    print("\nClases detectadas:")
    for i, clase in enumerate(nombres_clases):
        cantidad = np.sum(etiquetas == i)
        print(f"{i:02d}. {clase:20s} -> {cantidad} audios")

    (
        archivos_train,
        etiquetas_train,
        archivos_val,
        etiquetas_val,
        archivos_test,
        etiquetas_test
    ) = dividir_dataset(archivos, etiquetas)

    print("\nDivisión del dataset:")
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

    for secuencias, _ in dataset_train.take(1):
        input_shape = secuencias.shape[1:]
        break

    print(f"\nForma de entrada del modelo GRU: {input_shape}")
    print("Interpretación: pasos de tiempo x características de frecuencia")

    modelo = crear_modelo_gru(
        input_shape=input_shape,
        numero_clases=len(nombres_clases)
    )

    print("\nResumen del modelo GRU:")
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "domotica_gru.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nIniciando entrenamiento GRU...\n")

    historial = modelo.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nGuardando modelo GRU final...")

    modelo.save(MODELS_DIR / "domotica_gru.keras")

    with open(MODELS_DIR / "class_names_gru.json", "w", encoding="utf-8") as archivo_json:
        json.dump(
            nombres_clases,
            archivo_json,
            indent=2,
            ensure_ascii=False
        )

    guardar_grafica_entrenamiento(
        historial,
        MODELS_DIR / "training_accuracy_gru.png"
    )

    guardar_grafica_perdida(
        historial,
        MODELS_DIR / "training_loss_gru.png"
    )

    evaluar_modelo(
        modelo,
        dataset_test,
        nombres_clases
    )

    print("\nEntrenamiento GRU terminado.")
    print("\nArchivos generados:")
    print(f"- {MODELS_DIR / 'domotica_gru.keras'}")
    print(f"- {MODELS_DIR / 'class_names_gru.json'}")
    print(f"- {MODELS_DIR / 'training_accuracy_gru.png'}")
    print(f"- {MODELS_DIR / 'training_loss_gru.png'}")
    print(f"- {MODELS_DIR / 'confusion_matrix_gru.png'}")


if __name__ == "__main__":
    main()