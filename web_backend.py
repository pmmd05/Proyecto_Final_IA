import json
import time
import queue
import threading
import asyncio
import csv
from collections import deque
from pathlib import Path
from datetime import datetime

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
    MEL_BINS,
    LOWER_EDGE_HERTZ,
    UPPER_EDGE_HERTZ,
)


# =========================================================
# MODELO FINAL: GRU + LOG MEL-SPECTROGRAM
# =========================================================

MODEL_PATH = MODELS_DIR / "domotica_gru_mel.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names_gru_mel.json"

BACKGROUND_CLASS = "RUIDO_FONDO"


# =========================================================
# ARCHIVO PARA GUARDAR LATENCIAS
# =========================================================

LATENCY_LOG_PATH = Path("latency_results.csv")


# =========================================================
# CONFIGURACIÓN DEL VAD
# =========================================================

BLOCK_SECONDS = 0.05
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)

PRE_ROLL_SECONDS = 0.35
PRE_ROLL_BLOCKS = int(PRE_ROLL_SECONDS / BLOCK_SECONDS)

BASE_ENERGY_THRESHOLD = 0.020

# Si necesitas forzar manualmente:
# Ejemplo: 0.12, 0.18, 0.22
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
    loop = getattr(app.state, "loop", None)

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
# SERVICIO PRINCIPAL DE VOZ
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
            raise FileNotFoundError(f"No se encontró el archivo de clases: {CLASS_NAMES_PATH}")

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            self.class_names = json.load(file)

        if BACKGROUND_CLASS not in self.class_names:
            raise ValueError(f"No se encontró la clase {BACKGROUND_CLASS}")

        self.ensure_latency_csv()

        emit_event({
            "type": "status",
            "message": "Modelo GRU + Mel cargado correctamente."
        })

        emit_event({
            "type": "classes",
            "classes": self.class_names
        })

    # -----------------------------------------------------
    # CSV de latencia
    # -----------------------------------------------------

    def ensure_latency_csv(self):
        if not LATENCY_LOG_PATH.exists():
            with open(LATENCY_LOG_PATH, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "predicted_class",
                    "confidence",
                    "accepted",
                    "arduino_command",
                    "vad_ms",
                    "preprocess_ms",
                    "inference_ms",
                    "decision_ms",
                    "serial_ms",
                    "ack_ms",
                    "total_ms",
                    "fps_inference",
                    "fps_total"
                ])

    def save_latency_row(
        self,
        predicted_class,
        confidence,
        accepted,
        arduino_cmd,
        vad_ms,
        preprocess_ms,
        inference_ms,
        decision_ms,
        serial_ms,
        ack_ms,
        total_ms,
        fps_inference,
        fps_total
    ):
        with open(LATENCY_LOG_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                predicted_class,
                f"{confidence:.4f}",
                accepted,
                arduino_cmd,
                f"{vad_ms:.2f}",
                f"{preprocess_ms:.2f}",
                f"{inference_ms:.2f}",
                f"{decision_ms:.2f}",
                f"{serial_ms:.2f}",
                f"{ack_ms:.2f}",
                f"{total_ms:.2f}",
                f"{fps_inference:.2f}",
                f"{fps_total:.2f}",
            ])

    # -----------------------------------------------------
    # Arduino
    # -----------------------------------------------------

    def connect_arduino(self, port: str):
        if self.serial_connection:
            self.serial_connection.close()

        self.serial_connection = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)

        # Limpiar mensajes iniciales del Arduino
        self.clear_serial_buffer()

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

    def clear_serial_buffer(self):
        if self.serial_connection is None:
            return

        try:
            while self.serial_connection.in_waiting > 0:
                self.serial_connection.readline()
        except Exception:
            pass

    def send_to_arduino_with_ack(self, command: str):
        """
        Envía comando al Arduino y mide:
        - tiempo de escritura Serial
        - tiempo hasta recibir ACK

        Retorna:
        serial_ms, ack_ms, ack_text
        """

        if not self.use_arduino:
            emit_event({
                "type": "log",
                "message": f"Arduino desactivado. Comando no enviado: {command}"
            })
            return 0.0, 0.0, "ARDUINO_DISABLED"

        if self.serial_connection is None:
            emit_event({
                "type": "log",
                "message": "No se envió a Arduino: no está conectado."
            })
            return 0.0, 0.0, "ARDUINO_NOT_CONNECTED"

        try:
            self.clear_serial_buffer()

            t_serial_start = time.perf_counter()

            self.serial_connection.write((command + "\n").encode("utf-8"))
            self.serial_connection.flush()

            t_serial_end = time.perf_counter()

            serial_ms = (t_serial_end - t_serial_start) * 1000.0

            # Esperar ACK del Arduino
            ack_text = ""
            t_ack_start = time.perf_counter()
            ack_timeout_seconds = 1.5

            while time.perf_counter() - t_ack_start < ack_timeout_seconds:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode("utf-8", errors="ignore").strip()

                    if line:
                        emit_event({
                            "type": "log",
                            "message": f"Arduino responde: {line}"
                        })

                        # Tomamos como ACK cualquier línea que empiece con ACK_
                        if line.startswith("ACK_"):
                            ack_text = line
                            break

            t_ack_end = time.perf_counter()

            ack_ms = (t_ack_end - t_serial_end) * 1000.0 if ack_text else 0.0

            if ack_text:
                emit_event({
                    "type": "log",
                    "message": f"ACK recibido: {ack_text} | ACK latency: {ack_ms:.2f} ms"
                })
            else:
                emit_event({
                    "type": "log",
                    "message": "No se recibió ACK dentro del tiempo esperado."
                })

            emit_event({
                "type": "log",
                "message": f"Enviado a Arduino: {command} | Serial: {serial_ms:.2f} ms"
            })

            return serial_ms, ack_ms, ack_text

        except Exception as e:
            emit_event({
                "type": "error",
                "message": f"Error enviando a Arduino: {e}"
            })
            return 0.0, 0.0, "ERROR"

    def send_manual_command(self, command: str):
        serial_ms, ack_ms, ack_text = self.send_to_arduino_with_ack(command)

        emit_event({
            "type": "log",
            "message": f"Prueba manual enviada: {command} | Serial: {serial_ms:.2f} ms | ACK: {ack_ms:.2f} ms | {ack_text}"
        })

    # -----------------------------------------------------
    # Audio
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

    # -----------------------------------------------------
    # VAD
    # -----------------------------------------------------

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
            emit_event({
                "type": "log",
                "message": f"Umbral VAD manual usado: {MANUAL_ENERGY_THRESHOLD:.5f}"
            })
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

            has_voice, energy, zcr = self.block_has_voice(
                block,
                energy_threshold
            )

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
    # Log Mel-Spectrogram
    # -----------------------------------------------------

    def convert_to_log_mel_spectrogram(self, audio):
        audio = self.adjust_audio_duration(audio)
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

        mean = tf.reduce_mean(log_mel_spectrogram)
        std = tf.math.reduce_std(log_mel_spectrogram)

        log_mel_spectrogram = (
            log_mel_spectrogram - mean
        ) / (std + 1e-6)

        sequence = log_mel_spectrogram[tf.newaxis, ...]

        return sequence

    # -----------------------------------------------------
    # Inicio / detener escucha
    # -----------------------------------------------------

    def start(self):
        if self.is_listening:
            emit_event({
                "type": "log",
                "message": "La escucha ya estaba iniciada."
            })
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

    # -----------------------------------------------------
    # Loop principal con medición de latencia
    # -----------------------------------------------------

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

                    # =============================================
                    # 1. Captura + VAD
                    # =============================================

                    t_start_total = time.perf_counter()

                    audio = self.get_next_utterance(self.energy_threshold)

                    t_end_vad = time.perf_counter()

                    if audio is None:
                        continue

                    emit_event({
                        "type": "status",
                        "message": "Clasificando comando..."
                    })

                    # =============================================
                    # 2. Preprocesamiento + Log Mel
                    # =============================================

                    t_start_preprocess = time.perf_counter()

                    sequence = self.convert_to_log_mel_spectrogram(audio)

                    t_end_preprocess = time.perf_counter()

                    # =============================================
                    # 3. Inferencia GRU
                    # =============================================

                    predictions = self.model.predict(sequence, verbose=0)[0]

                    t_end_inference = time.perf_counter()

                    # =============================================
                    # 4. Decisión del backend
                    # =============================================

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

                    action = ACTION_LABELS.get(predicted_class, predicted_class)
                    arduino_cmd = ARDUINO_COMMANDS.get(predicted_class, "--")

                    accepted = (
                        predicted_class != BACKGROUND_CLASS
                        and confidence >= COMMAND_CONFIDENCE_THRESHOLD
                    )

                    t_end_decision = time.perf_counter()

                    serial_ms = 0.0
                    ack_ms = 0.0
                    ack_text = ""

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
                            serial_ms, ack_ms, ack_text = self.send_to_arduino_with_ack(arduino_cmd)

                    t_end_total = time.perf_counter()

                    # =============================================
                    # Cálculo de latencias
                    # =============================================

                    vad_ms = (t_end_vad - t_start_total) * 1000.0
                    preprocess_ms = (t_end_preprocess - t_start_preprocess) * 1000.0
                    inference_ms = (t_end_inference - t_end_preprocess) * 1000.0
                    decision_ms = (t_end_decision - t_end_inference) * 1000.0
                    total_ms = (t_end_total - t_start_total) * 1000.0

                    fps_inference = 1000.0 / inference_ms if inference_ms > 0 else 0.0
                    fps_total = 1000.0 / total_ms if total_ms > 0 else 0.0

                    latency_message = (
                        f"Latencia | "
                        f"VAD: {vad_ms:.2f} ms | "
                        f"Preproc: {preprocess_ms:.2f} ms | "
                        f"Inferencia: {inference_ms:.2f} ms | "
                        f"Decisión: {decision_ms:.2f} ms | "
                        f"Serial: {serial_ms:.2f} ms | "
                        f"ACK: {ack_ms:.2f} ms | "
                        f"Total: {total_ms:.2f} ms | "
                        f"FPS inferencia: {fps_inference:.2f} | "
                        f"FPS total: {fps_total:.2f}"
                    )

                    emit_event({
                        "type": "log",
                        "message": latency_message
                    })

                    emit_event({
                        "type": "latency",
                        "vad_ms": vad_ms,
                        "preprocess_ms": preprocess_ms,
                        "inference_ms": inference_ms,
                        "decision_ms": decision_ms,
                        "serial_ms": serial_ms,
                        "ack_ms": ack_ms,
                        "total_ms": total_ms,
                        "fps_inference": fps_inference,
                        "fps_total": fps_total
                    })

                    self.save_latency_row(
                        predicted_class=predicted_class,
                        confidence=confidence,
                        accepted=accepted,
                        arduino_cmd=arduino_cmd,
                        vad_ms=vad_ms,
                        preprocess_ms=preprocess_ms,
                        inference_ms=inference_ms,
                        decision_ms=decision_ms,
                        serial_ms=serial_ms,
                        ack_ms=ack_ms,
                        total_ms=total_ms,
                        fps_inference=fps_inference,
                        fps_total=fps_total
                    )

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
# MODELOS DE PETICIÓN API
# =========================================================

class ArduinoConnectRequest(BaseModel):
    port: str


class UseArduinoRequest(BaseModel):
    enabled: bool


class ManualCommandRequest(BaseModel):
    command: str


# =========================================================
# RUTAS API
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
    voice_service.send_manual_command(request.command)

    return {
        "ok": True,
        "command": request.command
    }