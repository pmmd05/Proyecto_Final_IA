from pathlib import Path

# =========================================================
# RUTAS PRINCIPALES DEL PROYECTO
# =========================================================

DATASET_DIR = Path("dataset_domotica_voz")
MODELS_DIR = Path("models")


# =========================================================
# CONFIGURACIÓN DE AUDIO
# =========================================================

# Frecuencia de muestreo.
# 16000 Hz es suficiente para comandos de voz cortos.
SAMPLE_RATE = 16000

# Duración de cada grabación en segundos.
# Cada comando será grabado como un audio corto de 2 segundos.
DURATION_SECONDS = 2.0

# Cantidad total de muestras por audio.
AUDIO_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)


# =========================================================
# CONFIGURACIÓN DEL ESPECTROGRAMA
# =========================================================

# Tamaño de ventana para STFT.
FRAME_LENGTH = 256

# Desplazamiento entre ventanas.
FRAME_STEP = 128


# =========================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =========================================================

BATCH_SIZE = 32
EPOCHS = 35

VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42


# =========================================================
# UMBRAL PARA ACEPTAR PREDICCIONES EN VIVO
# =========================================================

CONFIDENCE_THRESHOLD = 0.75


# =========================================================
# CLASES DEL MODELO
# =========================================================

COMMANDS = {
    "LUCES_ON": "enciende las luces",
    "LUCES_OFF": "apaga las luces",

    "AIRE_ON": "enciende el aire",
    "AIRE_OFF": "apaga el aire",

    "BOMBA_REGAR": "riega las plantas",

    "RUIDO_FONDO": "ruido, silencio o habla no relacionada"
}


# =========================================================
# MAPEO DE CLASES DEL MODELO A COMANDOS PARA ARDUINO
# =========================================================

ARDUINO_COMMANDS = {
    "LUCES_ON": "LUZ_ON",
    "LUCES_OFF": "LUZ_OFF",

    "AIRE_ON": "AIRE_ON",
    "AIRE_OFF": "AIRE_OFF",

    "BOMBA_REGAR": "BOMBA_REGAR",
}