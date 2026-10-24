import cv2
import os
from pathlib import Path

def difuminar_rostros_carpeta(carpeta_entrada, carpeta_salida):
    # Cargar el clasificador preentrenado de rostros
    print("Iniciando Funcion...")
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Extensiones de imagen soportadas
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Contadores para reporte
    procesadas = 0
    total_rostros = 0

    # Recorrer las imágenes de la carpeta de entrada
    archivos_imagen = [
        archivo
        for archivo in os.listdir(carpeta_entrada)
        if os.path.isfile(os.path.join(carpeta_entrada, archivo))
        and Path(archivo).suffix.lower() in extensiones_validas
    ]
    total_imagenes = len(archivos_imagen)

    for indice, archivo in enumerate(archivos_imagen, start=1):
        ruta_archivo = os.path.join(carpeta_entrada, archivo)

        print(f"[{indice}/{total_imagenes}] Procesando: {archivo}", flush=True)
        imagen = cv2.imread(ruta_archivo)
        if imagen is None:
            print(f"Error al cargar {archivo}, se omite.", flush=True)
            continue

        # Convertir a escala de grises para la detección
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        rostros = face_cascade.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Aplicar difuminado
        for (x, y, w, h) in rostros:
            region_rostro = imagen[y:y+h, x:x+w]
            rostro_difuminado = cv2.GaussianBlur(region_rostro, (99, 99), 30)
            imagen[y:y+h, x:x+w] = rostro_difuminado

        # Guardar la imagen con el mismo nombre en la carpeta de salida
        ruta_guardado = os.path.join(carpeta_salida, archivo)
        cv2.imwrite(ruta_guardado, imagen)

        procesadas += 1
        total_rostros += len(rostros)
        print(f"Lista: {archivo} | Rostros difuminados: {len(rostros)}", flush=True)

    print("\n--- Resumen ---")
    print(f"Total imágenes procesadas: {procesadas}")
    print(f"Total rostros difuminados: {total_rostros}")
    print(f"Imágenes guardadas en: {carpeta_salida}")

# Ejemplo de uso:
difuminar_rostros_carpeta(r'C:\Users\Emman\Desktop\Proyecto IA\A PROCESAR\debitho.07', r'C:\Users\Emman\Desktop\Proyecto IA\imagenes_procesadas/debitho.07/')