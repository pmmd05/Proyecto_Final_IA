import json
import time
import queue
import threading
import asyncio
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import tensorflow as tf
import serial

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
# CONFIGURACIÓN DEL MODELO GRU
# =========================================================

MODEL_PATH = MODELS_DIR / "domotica_gru.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names_gru.json"

BACKGROUND_CLASS = "RUIDO_FONDO"


# =========================================================
# CONFIGURACIÓN DEL VAD
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
MAX_UTTERANCE_SECONDS = 3.00

COOLDOWN_SECONDS = 0.50

COMMAND_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD


ACTION_LABELS = {
    "LUCES_ON": "ENCENDER LUCES",
    "LUCES_OFF": "APAGAR LUCES",
    "AIRE_ON": "ENCENDER AIRE",
    "AIRE_OFF": "APAGAR AIRE",
    "BOMBA_REGAR": "REGAR PLANTAS",
    "RUIDO_FONDO": "IGNORAR",
}


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="Panel de Domótica por Voz")

app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/")
def home():
    return FileResponse("web/index.html")


# =========================================================
# WEBSOCKET MANAGER
# =========================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


def emit_event(event: dict):
    """
    Permite enviar eventos desde el hilo del micrófono hacia el WebSocket.
    """
    loop = app.state.loop

    if loop is not None:
        asyncio.run_coroutine_threadsafe(manager.broadcast(event), loop)


@app.on_event("startup")
async def startup_event():
    app.state.loop = asyncio.get_running_loop()
    voice_service.load_model()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        await websocket.send_json({
            "type": "status",
            "message": "Interfaz conectada al backend."
        })

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =========================================================
# VOICE SERVICE
# =========================================================

class VoiceService:
    def __init__(self):
        self.model = None
        self.class_names = None

        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.stream = None

        self.is_listening = False
        self.energy_threshold = BASE_ENERGY_THRESHOLD

        self.serial_connection = None
        self.use_arduino = False

    # -----------------------------------------------------
    # Modelo
    # -----------------------------------------------------

    def load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"No se encontró class_names_gru.json: {CLASS_NAMES_PATH}")

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            self.class_names = json.load(file)

        emit_event({
            "type": "status",
            "message": "Modelo GRU cargado correctamente."
        })

        emit_event({
            "type": "classes",
            "classes": self.class_names
        })

    # -----------------------------------------------------
    # Arduino
    # -----------------------------------------------------

    def connect_arduino(self, port: str):
        if self.serial_connection:
            self.serial_connection.close()

        self.serial_connection = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)

        emit_event({
            "type": "arduino",
            "connected": True,
            "message": f"Arduino conectado en {port}."
        })

    def disconnect_arduino(self):
        if self.serial_connection:
            self.serial_connection.close()

        self.serial_connection = None

        emit_event({
            "type": "arduino",
            "connected": False,
            "message": "Arduino desconectado."
        })

    def send_to_arduino(self, command: str):
        if not self.use_arduino:
            return

        if self.serial_connection is None:
            emit_event({
                "type": "log",
                "message": "No se envió a Arduino: no está conectado."
            })
            return

        self.serial_connection.write((command + "\n").encode("utf-8"))

        emit_event({
            "type": "log",
            "message": f"Enviado a Arduino: {command}"
        })

    # -----------------------------------------------------
    # Audio callback
    # -----------------------------------------------------

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        self.audio_queue.put(indata.copy())

    def clear_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # -----------------------------------------------------
    # VAD helpers
    # -----------------------------------------------------

    def calculate_energy(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(audio))))

    def calculate_zcr(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.squeeze(audio)

        if len(audio) < 2:
            return 0.0

        crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        return float(crossings / len(audio))

    def normalize_audio(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.squeeze(audio)

        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

        if max_val > 0:
            audio = audio / max_val * 0.9

        return audio

    def adjust_audio_duration(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.squeeze(audio)

        if len(audio) < AUDIO_SAMPLES:
            missing = AUDIO_SAMPLES - len(audio)
            audio = np.pad(audio, (0, missing), mode="constant")
        else:
            audio = audio[:AUDIO_SAMPLES]

        return audio

    def block_has_voice(self, audio, energy_threshold):
        energy = self.calculate_energy(audio)
        zcr = self.calculate_zcr(audio)

        energy_ok = energy >= energy_threshold

        if USE_ZCR:
            zcr_ok = ZCR_MIN <= zcr <= ZCR_MAX
        else:
            zcr_ok = True

        return energy_ok and zcr_ok, energy, zcr

    def calibrate_noise(self, seconds=2.0):
        if MANUAL_ENERGY_THRESHOLD is not None:
            return MANUAL_ENERGY_THRESHOLD

        emit_event({
            "type": "status",
            "message": "Calibrando ruido ambiente. No hablar..."
        })

        samples = []
        start = time.time()

        while time.time() - start < seconds and not self.stop_event.is_set():
            try:
                block = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            block = np.squeeze(block)
            energy = self.calculate_energy(block)
            samples.append(energy)

        if not samples:
            return BASE_ENERGY_THRESHOLD

        samples = np.array(samples)

        noise_median = float(np.median(samples))
        noise_p90 = float(np.percentile(samples, 90))

        threshold = max(
            BASE_ENERGY_THRESHOLD,
            noise_median * 1.8,
            noise_p90 * 1.2
        )

        threshold = min(threshold, 0.25)

        emit_event({
            "type": "log",
            "message": f"Umbral VAD calibrado: {threshold:.5f}"
        })

        return threshold

    def get_next_utterance(self, energy_threshold):
        pre_buffer = deque(maxlen=PRE_ROLL_BLOCKS)

        speech_blocks = []
        speech_started = False

        consecutive_speech = 0
        consecutive_silence = 0

        while not self.stop_event.is_set():
            try:
                block = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            block = np.squeeze(block)

            has_voice, energy, zcr = self.block_has_voice(block, energy_threshold)

            emit_event({
                "type": "energy",
                "energy": energy,
                "zcr": zcr
            })

            if not speech_started:
                pre_buffer.append(block)

                if has_voice:
                    consecutive_speech += 1
                else:
                    consecutive_speech = 0

                if consecutive_speech >= MIN_SPEECH_BLOCKS:
                    speech_started = True
                    speech_blocks = list(pre_buffer)
                    consecutive_silence = 0

                    emit_event({
                        "type": "status",
                        "message": "Voz detectada. Segmentando frase..."
                    })

                    emit_event({
                        "type": "log",
                        "message": "Inicio de voz detectado."
                    })

            else:
                speech_blocks.append(block)

                if has_voice:
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1

                current_duration = len(speech_blocks) * BLOCK_SECONDS

                if consecutive_silence >= END_SILENCE_BLOCKS:
                    audio = np.concatenate(speech_blocks)
                    audio = self.normalize_audio(audio)

                    duration = len(audio) / SAMPLE_RATE

                    emit_event({
                        "type": "log",
                        "message": f"Fin de frase detectado. Duración: {duration:.2f} s."
                    })

                    if duration < MIN_UTTERANCE_SECONDS:
                        emit_event({
                            "type": "log",
                            "message": "Frase demasiado corta. Ignorada."
                        })
                        return None

                    return audio

                if current_duration >= MAX_UTTERANCE_SECONDS:
                    audio = np.concatenate(speech_blocks)
                    audio = self.normalize_audio(audio)

                    duration = len(audio) / SAMPLE_RATE

                    emit_event({
                        "type": "log",
                        "message": f"Frase larga recortada. Duración: {duration:.2f} s."
                    })

                    return audio

        return None

    # -----------------------------------------------------
    # GRU preprocessing
    # -----------------------------------------------------

    def convert_to_spectral_sequence(self, audio):
        audio = self.adjust_audio_duration(audio)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        spectrogram = tf.signal.stft(
            audio,
            frame_length=FRAME_LENGTH,
            frame_step=FRAME_STEP
        )

        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.log(spectrogram + 1e-6)

        mean = tf.reduce_mean(spectrogram)
        std = tf.math.reduce_std(spectrogram)

        spectrogram = (spectrogram - mean) / (std + 1e-6)

        sequence = spectrogram[tf.newaxis, ...]

        return sequence

    def predict_audio(self, audio):
        sequence = self.convert_to_spectral_sequence(audio)

        predictions = self.model.predict(sequence, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = self.class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        top_indices = np.argsort(predictions)[::-1][:3]

        top3 = [
            {
                "class": self.class_names[i],
                "confidence": float(predictions[i])
            }
            for i in top_indices
        ]

        return predicted_class, confidence, top3

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------

    def start(self):
        if self.is_listening:
            return

        self.stop_event.clear()
        self.clear_audio_queue()

        self.worker_thread = threading.Thread(
            target=self.listen_loop,
            daemon=True
        )

        self.worker_thread.start()
        self.is_listening = True

    def stop(self):
        self.stop_event.set()

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception:
            pass

        self.is_listening = False

        emit_event({
            "type": "status",
            "message": "Escucha detenida."
        })

    def listen_loop(self):
        try:
            emit_event({
                "type": "status",
                "message": "Iniciando micrófono..."
            })

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=self.audio_callback
            ):
                self.energy_threshold = self.calibrate_noise(seconds=2.0)
                self.clear_audio_queue()

                emit_event({
                    "type": "status",
                    "message": "Escuchando continuamente..."
                })

                while not self.stop_event.is_set():
                    audio = self.get_next_utterance(self.energy_threshold)

                    if audio is None:
                        continue

                    emit_event({
                        "type": "status",
                        "message": "Clasificando comando..."
                    })

                    predicted_class, confidence, top3 = self.predict_audio(audio)

                    action = ACTION_LABELS.get(predicted_class, predicted_class)
                    arduino_cmd = ARDUINO_COMMANDS.get(predicted_class, "--")

                    accepted = (
                        predicted_class != BACKGROUND_CLASS
                        and confidence >= COMMAND_CONFIDENCE_THRESHOLD
                    )

                    emit_event({
                        "type": "prediction",
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "action": action,
                        "arduino_command": arduino_cmd,
                        "accepted": accepted,
                        "top3": top3
                    })

                    if predicted_class == BACKGROUND_CLASS:
                        emit_event({
                            "type": "log",
                            "message": "RUIDO_FONDO detectado. Se ignora."
                        })

                    elif confidence < COMMAND_CONFIDENCE_THRESHOLD:
                        emit_event({
                            "type": "log",
                            "message": f"Comando rechazado por baja confianza: {predicted_class} ({confidence * 100:.2f}%)."
                        })

                    else:
                        emit_event({
                            "type": "log",
                            "message": f"Comando aceptado: {predicted_class} | {confidence * 100:.2f}% | Arduino: {arduino_cmd}"
                        })

                        if arduino_cmd != "--":
                            self.send_to_arduino(arduino_cmd)

                    emit_event({
                        "type": "status",
                        "message": "Escuchando continuamente..."
                    })

                    time.sleep(COOLDOWN_SECONDS)

        except Exception as e:
            emit_event({
                "type": "error",
                "message": str(e)
            })

        finally:
            self.is_listening = False


voice_service = VoiceService()


# =========================================================
# API MODELS
# =========================================================

class ArduinoConnectRequest(BaseModel):
    port: str


class UseArduinoRequest(BaseModel):
    enabled: bool


class ManualCommandRequest(BaseModel):
    command: str


# =========================================================
# API ROUTES
# =========================================================

@app.post("/api/start")
def start_listening():
    voice_service.start()

    return {
        "ok": True,
        "message": "Escucha iniciada."
    }


@app.post("/api/stop")
def stop_listening():
    voice_service.stop()

    return {
        "ok": True,
        "message": "Escucha detenida."
    }


@app.post("/api/arduino/connect")
def connect_arduino(request: ArduinoConnectRequest):
    voice_service.connect_arduino(request.port)

    return {
        "ok": True,
        "message": f"Arduino conectado en {request.port}."
    }


@app.post("/api/arduino/disconnect")
def disconnect_arduino():
    voice_service.disconnect_arduino()

    return {
        "ok": True,
        "message": "Arduino desconectado."
    }


@app.post("/api/arduino/use")
def use_arduino(request: UseArduinoRequest):
    voice_service.use_arduino = request.enabled

    return {
        "ok": True,
        "enabled": request.enabled
    }


@app.post("/api/arduino/manual")
def manual_command(request: ManualCommandRequest):
    voice_service.send_to_arduino(request.command)

    return {
        "ok": True,
        "command": request.command
    }