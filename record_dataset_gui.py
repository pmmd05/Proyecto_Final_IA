import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import COMMANDS, SAMPLE_RATE


BASE_DATASET_DIR = Path("dataset_domotica_voz")


class GrabadorDatasetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recolector de Audios - Domótica por Voz")
        self.root.geometry("760x560")
        self.root.resizable(False, False)

        self.audio_chunks = []
        self.stream = None
        self.is_recording = False
        self.start_time = None

        self.class_names = list(COMMANDS.keys())
        self.current_class_index = 0

        self.crear_carpetas()
        self.crear_interfaz()
        self.actualizar_clase_actual()
        self.actualizar_contador()

    # =====================================================
    # CREACIÓN DE CARPETAS
    # =====================================================

    def crear_carpetas(self):
        BASE_DATASET_DIR.mkdir(exist_ok=True)

        for etiqueta in COMMANDS:
            carpeta = BASE_DATASET_DIR / etiqueta
            carpeta.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # INTERFAZ GRÁFICA
    # =====================================================

    def crear_interfaz(self):
        titulo = tk.Label(
            self.root,
            text="Recolector de Audios para Panel de Domótica por Voz",
            font=("Arial", 18, "bold")
        )
        titulo.pack(pady=15)

        descripcion = tk.Label(
            self.root,
            text="Presiona 'Iniciar grabación', di la frase indicada y luego presiona 'Detener y guardar'.",
            font=("Arial", 11)
        )
        descripcion.pack(pady=5)

        frame_datos = tk.LabelFrame(self.root, text="Datos de la grabación", padx=15, pady=10)
        frame_datos.pack(fill="x", padx=25, pady=10)

        tk.Label(frame_datos, text="ID Persona:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.persona_entry = tk.Entry(frame_datos, width=15)
        self.persona_entry.insert(0, "p01")
        self.persona_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame_datos, text="Entorno:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.entorno_entry = tk.Entry(frame_datos, width=15)
        self.entorno_entry.insert(0, "e01")
        self.entorno_entry.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(frame_datos, text="Meta por clase:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.meta_entry = tk.Entry(frame_datos, width=10)
        self.meta_entry.insert(0, "8")
        self.meta_entry.grid(row=0, column=5, padx=5, pady=5)

        frame_clase = tk.LabelFrame(self.root, text="Comando actual", padx=15, pady=15)
        frame_clase.pack(fill="x", padx=25, pady=10)

        tk.Label(frame_clase, text="Clase:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.clase_combo = ttk.Combobox(
            frame_clase,
            values=self.class_names,
            state="readonly",
            width=28
        )
        self.clase_combo.grid(row=0, column=1, padx=5, pady=5)
        self.clase_combo.current(0)
        self.clase_combo.bind("<<ComboboxSelected>>", self.cambiar_clase_combo)

        self.frase_label = tk.Label(
            frame_clase,
            text="",
            font=("Arial", 20, "bold"),
            fg="#1f4e79",
            wraplength=680,
            justify="center"
        )
        self.frase_label.grid(row=1, column=0, columnspan=4, padx=10, pady=20)

        self.contador_label = tk.Label(
            frame_clase,
            text="",
            font=("Arial", 12, "bold")
        )
        self.contador_label.grid(row=2, column=0, columnspan=4, pady=5)

        frame_estado = tk.Frame(self.root)
        frame_estado.pack(fill="x", padx=25, pady=10)

        self.estado_label = tk.Label(
            frame_estado,
            text="Estado: listo para grabar",
            font=("Arial", 13),
            fg="green"
        )
        self.estado_label.pack()

        self.tiempo_label = tk.Label(
            frame_estado,
            text="Duración actual: 0.0 s",
            font=("Arial", 12)
        )
        self.tiempo_label.pack(pady=5)

        frame_botones = tk.Frame(self.root)
        frame_botones.pack(pady=15)

        self.btn_iniciar = tk.Button(
            frame_botones,
            text="Iniciar grabación",
            font=("Arial", 13, "bold"),
            width=20,
            bg="#4CAF50",
            fg="white",
            command=self.iniciar_grabacion
        )
        self.btn_iniciar.grid(row=0, column=0, padx=10, pady=5)

        self.btn_detener = tk.Button(
            frame_botones,
            text="Detener y guardar",
            font=("Arial", 13, "bold"),
            width=20,
            bg="#D9534F",
            fg="white",
            state="disabled",
            command=self.detener_y_guardar
        )
        self.btn_detener.grid(row=0, column=1, padx=10, pady=5)

        self.btn_anterior = tk.Button(
            frame_botones,
            text="Clase anterior",
            font=("Arial", 11),
            width=18,
            command=self.clase_anterior
        )
        self.btn_anterior.grid(row=1, column=0, padx=10, pady=8)

        self.btn_siguiente = tk.Button(
            frame_botones,
            text="Siguiente clase",
            font=("Arial", 11),
            width=18,
            command=self.siguiente_clase
        )
        self.btn_siguiente.grid(row=1, column=1, padx=10, pady=8)

        frame_info = tk.LabelFrame(self.root, text="Recomendaciones", padx=15, pady=10)
        frame_info.pack(fill="x", padx=25, pady=10)

        recomendaciones = (
            "• Presiona iniciar, espera medio segundo y luego habla.\n"
            "• Di la frase con voz normal.\n"
            "• Evita cortar la grabación antes de terminar la frase.\n"
            "• Para RUIDO_FONDO puedes grabar silencio, ruido ambiente o habla no relacionada.\n"
            "• Usa e01 para entorno silencioso y e02 para entorno con ruido moderado."
        )

        tk.Label(
            frame_info,
            text=recomendaciones,
            justify="left",
            font=("Arial", 10)
        ).pack(anchor="w")

    # =====================================================
    # CONTROL DE CLASES
    # =====================================================

    def cambiar_clase_combo(self, event=None):
        seleccion = self.clase_combo.get()
        self.current_class_index = self.class_names.index(seleccion)
        self.actualizar_clase_actual()
        self.actualizar_contador()

    def actualizar_clase_actual(self):
        clase = self.class_names[self.current_class_index]
        frase = COMMANDS[clase]

        self.clase_combo.current(self.current_class_index)

        if clase == "RUIDO_FONDO":
            texto = "Graba silencio, ruido ambiente o habla no relacionada"
        else:
            texto = f'Di: "{frase}"'

        self.frase_label.config(text=texto)

    def siguiente_clase(self):
        if self.is_recording:
            messagebox.showwarning("Grabación activa", "Primero detén la grabación actual.")
            return

        self.current_class_index = (self.current_class_index + 1) % len(self.class_names)
        self.actualizar_clase_actual()
        self.actualizar_contador()

    def clase_anterior(self):
        if self.is_recording:
            messagebox.showwarning("Grabación activa", "Primero detén la grabación actual.")
            return

        self.current_class_index = (self.current_class_index - 1) % len(self.class_names)
        self.actualizar_clase_actual()
        self.actualizar_contador()

    # =====================================================
    # GRABACIÓN
    # =====================================================

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        self.audio_chunks.append(indata.copy())

    def iniciar_grabacion(self):
        persona_id = self.persona_entry.get().strip()
        entorno_id = self.entorno_entry.get().strip()

        if not persona_id:
            messagebox.showerror("Error", "Debes escribir un ID de persona, por ejemplo p01.")
            return

        if not entorno_id:
            messagebox.showerror("Error", "Debes escribir un ID de entorno, por ejemplo e01.")
            return

        if self.is_recording:
            return

        self.audio_chunks = []

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self.audio_callback
            )

            self.stream.start()
            self.is_recording = True
            self.start_time = time.time()

            self.estado_label.config(text="Estado: grabando...", fg="red")
            self.btn_iniciar.config(state="disabled")
            self.btn_detener.config(state="normal")

            self.actualizar_tiempo()

        except Exception as e:
            messagebox.showerror("Error al iniciar grabación", str(e))

    def detener_y_guardar(self):
        if not self.is_recording:
            return

        try:
            self.stream.stop()
            self.stream.close()
            self.stream = None

            self.is_recording = False

            if len(self.audio_chunks) == 0:
                messagebox.showerror("Error", "No se capturó audio.")
                self.restaurar_botones()
                return

            audio = np.concatenate(self.audio_chunks, axis=0)
            audio = np.squeeze(audio)

            duracion = len(audio) / SAMPLE_RATE

            if duracion < 0.5:
                messagebox.showwarning(
                    "Audio muy corto",
                    "La grabación duró menos de 0.5 segundos. Intenta grabar de nuevo."
                )
                self.restaurar_botones()
                return

            ruta_salida = self.generar_ruta_salida()
            self.guardar_audio(ruta_salida, audio)

            self.estado_label.config(
                text=f"Guardado: {ruta_salida.name}",
                fg="green"
            )

            self.restaurar_botones()
            self.actualizar_contador()

            meta = self.obtener_meta()
            grabadas = self.contar_grabaciones_actuales()

            if grabadas >= meta:
                messagebox.showinfo(
                    "Meta alcanzada",
                    f"Ya alcanzaste la meta de {meta} muestras para esta clase."
                )

        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))
            self.restaurar_botones()

    def guardar_audio(self, ruta_salida, audio):
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0

        if max_val > 0:
            audio = audio / max_val * 0.9

        sf.write(str(ruta_salida), audio, SAMPLE_RATE)

    def restaurar_botones(self):
        self.btn_iniciar.config(state="normal")
        self.btn_detener.config(state="disabled")
        self.tiempo_label.config(text="Duración actual: 0.0 s")

    def actualizar_tiempo(self):
        if self.is_recording and self.start_time is not None:
            duracion = time.time() - self.start_time
            self.tiempo_label.config(text=f"Duración actual: {duracion:.1f} s")
            self.root.after(100, self.actualizar_tiempo)

    # =====================================================
    # ARCHIVOS Y CONTADORES
    # =====================================================

    def generar_ruta_salida(self):
        persona_id = self.persona_entry.get().strip()
        entorno_id = self.entorno_entry.get().strip()
        clase = self.class_names[self.current_class_index]

        carpeta = BASE_DATASET_DIR / clase
        carpeta.mkdir(parents=True, exist_ok=True)

        numero = self.obtener_siguiente_numero(carpeta, persona_id, entorno_id)

        nombre_archivo = f"{persona_id}_{entorno_id}_r{numero:02d}.wav"

        return carpeta / nombre_archivo

    def obtener_siguiente_numero(self, carpeta, persona_id, entorno_id):
        patron = f"{persona_id}_{entorno_id}_r*.wav"
        archivos = list(carpeta.glob(patron))

        numeros = []

        for archivo in archivos:
            nombre = archivo.stem
            try:
                parte_r = nombre.split("_")[-1]
                numero = int(parte_r.replace("r", ""))
                numeros.append(numero)
            except ValueError:
                pass

        if len(numeros) == 0:
            return 1

        return max(numeros) + 1

    def contar_grabaciones_actuales(self):
        persona_id = self.persona_entry.get().strip()
        entorno_id = self.entorno_entry.get().strip()
        clase = self.class_names[self.current_class_index]

        carpeta = BASE_DATASET_DIR / clase

        if not carpeta.exists():
            return 0

        patron = f"{persona_id}_{entorno_id}_r*.wav"
        return len(list(carpeta.glob(patron)))

    def actualizar_contador(self):
        clase = self.class_names[self.current_class_index]
        grabadas = self.contar_grabaciones_actuales()
        meta = self.obtener_meta()

        self.contador_label.config(
            text=f"Clase: {clase} | Grabaciones actuales para esta persona/entorno: {grabadas}/{meta}"
        )

    def obtener_meta(self):
        try:
            meta = int(self.meta_entry.get().strip())
            if meta <= 0:
                return 1
            return meta
        except ValueError:
            return 1


def main():
    root = tk.Tk()
    app = GrabadorDatasetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()