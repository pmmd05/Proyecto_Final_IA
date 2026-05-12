from collections import defaultdict
from pathlib import Path

from config import DATASET_DIR, COMMANDS


def obtener_info_archivo(nombre_archivo):
    """
    Espera nombres como:
    p01_e01_r01.wav
    p02_e02_r15.wav

    Retorna:
    persona = p01
    entorno = e01
    """
    partes = nombre_archivo.stem.split("_")

    if len(partes) < 3:
        return None, None

    persona = partes[0]
    entorno = partes[1]

    return persona, entorno


def main():
    print("\n=== REVISIÓN DEL DATASET DE DOMÓTICA POR VOZ ===\n")

    total_general = 0
    total_por_persona = defaultdict(int)
    total_por_entorno = defaultdict(int)

    print("Conteo por clase:")
    print("-" * 55)

    for clase in COMMANDS.keys():
        carpeta = DATASET_DIR / clase

        if not carpeta.exists():
            cantidad = 0
            archivos = []
        else:
            archivos = list(carpeta.glob("*.wav"))
            cantidad = len(archivos)

        total_general += cantidad

        print(f"{clase:20s} | {cantidad:4d} muestras")

        for archivo in archivos:
            persona, entorno = obtener_info_archivo(archivo)

            if persona:
                total_por_persona[persona] += 1

            if entorno:
                total_por_entorno[entorno] += 1

    print("-" * 55)
    print(f"{'TOTAL GENERAL':20s} | {total_general:4d} muestras")

    print("\n=== Revisión de requisitos ===")

    if total_general >= 1500:
        print("✅ Cumple con mínimo de 1,500 muestras.")
    else:
        faltantes = 1500 - total_general
        print(f"❌ Aún faltan {faltantes} muestras para llegar a 1,500.")

    ruido_dir = DATASET_DIR / "RUIDO_FONDO"
    ruido_count = len(list(ruido_dir.glob("*.wav"))) if ruido_dir.exists() else 0

    if ruido_count >= 200:
        print("✅ Cumple con mínimo de 200 muestras de RUIDO_FONDO.")
    else:
        faltantes_ruido = 200 - ruido_count
        print(f"❌ Aún faltan {faltantes_ruido} muestras de RUIDO_FONDO.")

    print("\n=== Conteo por persona ===")
    if total_por_persona:
        for persona, cantidad in sorted(total_por_persona.items()):
            print(f"{persona:10s}: {cantidad:4d} muestras")
    else:
        print("No se detectaron personas en los nombres de archivo.")

    print("\n=== Conteo por entorno ===")
    if total_por_entorno:
        for entorno, cantidad in sorted(total_por_entorno.items()):
            print(f"{entorno:10s}: {cantidad:4d} muestras")
    else:
        print("No se detectaron entornos en los nombres de archivo.")

    print("\n=== Recomendaciones ===")

    if total_general < 1500:
        print("- Graba más muestras antes del entrenamiento final.")

    if ruido_count < 200:
        print("- Refuerza la clase RUIDO_FONDO. Es importante para evitar activaciones falsas.")

    print("- Intenta que todas las clases de comandos tengan cantidades similares.")
    print("- Verifica que existan muestras en al menos dos entornos: e01 y e02.")
    print("- Verifica que haya audios de integrantes y de al menos cinco voluntarios externos.")


if __name__ == "__main__":
    main()