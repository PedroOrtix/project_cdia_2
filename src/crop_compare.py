import cv2
import numpy as np
from PIL import Image
from .temperature_utils import calcular_temperatura, ajustar_temperatura

def recortar_imagen(lista_palabras, palabra, img_array, ancho=512, alto=512):
    """
    Recorta una sección de la imagen alrededor de una palabra específica.
    
    Args:
        lista_palabras (list): Lista de detecciones de palabras con sus coordenadas.
        palabra (str): Palabra a buscar en la imagen.
        img_array (numpy.ndarray): Imagen en formato array de numpy.
        ancho (int, opcional): Ancho deseado del recorte. Por defecto 512.
        alto (int, opcional): Alto deseado del recorte. Por defecto 512.
        
    Returns:
        tuple: Tupla conteniendo la imagen recortada y las coordenadas del recorte (x_min, y_min, x_max, y_max).
               Retorna (None, None) si no se encuentra la palabra.
    """
    altura_img, anchura_img, _ = img_array.shape
    mitad_ancho = ancho // 2
    mitad_alto = alto // 2
    
    es_paddle = isinstance(lista_palabras[0], list) and len(lista_palabras[0]) == 2
    
    for detection in lista_palabras:
        if es_paddle:
            coords = detection[0]
            texto = detection[1][0]
        else:
            coords = detection[0]
            texto = detection[1]
            
        if texto == palabra:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            x4, y4 = coords[3]
            
            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            x_min = int(x_center - mitad_ancho)
            x_max = int(x_center + mitad_ancho)
            y_min = int(y_center - mitad_alto)
            y_max = int(y_center + mitad_alto)
            
            if x_min < 0:
                x_min = 0
                x_max = min(ancho, anchura_img)
            if x_max > anchura_img:
                x_max = anchura_img
                x_min = max(0, anchura_img - ancho)
            if y_min < 0:
                y_min = 0
                y_max = min(alto, altura_img)
            if y_max > altura_img:
                y_max = altura_img
                y_min = max(0, altura_img - alto)
            
            imagen_recortada = img_array[y_min:y_max, x_min:x_max]
            return imagen_recortada, (x_min, y_min, x_max, y_max)
    return None, None

def comparar_imagenes(image_path1, image_path2, coordinates, save_path=False, adjust_temp=False):
    """
    Compara dos imágenes mostrándolas lado a lado, con opción de ajuste de temperatura.
    
    Args:
        image_path1 (str): Ruta de la primera imagen.
        image_path2 (str): Ruta de la segunda imagen.
        coordinates (list): Lista de coordenadas [x1, y1, x2, y2, x3, y3, x4, y4] para el recorte.
        save_path (str, opcional): Ruta donde guardar las imágenes comparadas. Por defecto False.
        adjust_temp (bool, opcional): Si se debe ajustar la temperatura de color. Por defecto False.
        
    Returns:
        PIL.Image: Imagen combinada con ambas imágenes lado a lado.
    """
    imagen_original = cv2.imread(image_path1)
    imagen_modificada = cv2.imread(image_path2)

    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates
    left = min(x1, x2, x3, x4)
    top = min(y1, y2, y3, y4)
    right = max(x1, x2, x3, x4)
    bottom = max(y1, y2, y3, y4)

    imagen_original_cropped = imagen_original[top:bottom, left:right]
    imagen_modificada_cropped = imagen_modificada[top:bottom, left:right]

    if adjust_temp:
        temp_original = calcular_temperatura(imagen_original_cropped)
        temp_modificada = calcular_temperatura(imagen_modificada_cropped)
        diferencia_temperatura = temp_original - temp_modificada
        imagen_ajustada = ajustar_temperatura(imagen_modificada_cropped, diferencia_temperatura)
        imagen_original_pil = Image.fromarray(cv2.cvtColor(imagen_original_cropped, cv2.COLOR_BGR2RGB))
        imagen_ajustada_pil = Image.fromarray(cv2.cvtColor(imagen_ajustada, cv2.COLOR_BGR2RGB))
    else:
        imagen_original_pil = Image.fromarray(cv2.cvtColor(imagen_original_cropped, cv2.COLOR_BGR2RGB))
        imagen_ajustada_pil = Image.fromarray(cv2.cvtColor(imagen_modificada_cropped, cv2.COLOR_BGR2RGB))

    width, height = imagen_original_pil.size
    comparison_image = Image.new('RGB', (2 * width, height))
    comparison_image.paste(imagen_original_pil, (0, 0))
    comparison_image.paste(imagen_ajustada_pil, (width, 0))

    if save_path:
        imagen_original_pil.save(f"{save_path}/{image_path1.split('/')[-1].split('.')[0]}_cropped.jpg")
        imagen_ajustada_pil.save(f"{save_path}/{image_path2.split('/')[-1].split('.')[0]}_cropped.jpg")
        comparison_image.save(f"{save_path}/comparison.jpg")

    return comparison_image

def reemplazar_parte_imagen(original_image_pil, modified_image_pil, coordinates, adjust_temp=False):
    """
    Reemplaza una sección de la imagen original con la correspondiente sección de la imagen modificada.
    
    Args:
        original_image_pil (PIL.Image): Imagen original.
        modified_image_pil (PIL.Image): Imagen modificada.
        coordinates (list): Lista de coordenadas [x1, y1, x2, y2, x3, y3, x4, y4] para el reemplazo.
        adjust_temp (bool, opcional): Si se debe ajustar la temperatura de color. Por defecto False.
        
    Returns:
        PIL.Image: Imagen combinada con la sección reemplazada.
    """
    original_image_cv2 = cv2.cvtColor(np.array(original_image_pil), cv2.COLOR_RGB2BGR)
    modified_image_cv2 = cv2.cvtColor(np.array(modified_image_pil), cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates
    left = min(x1, x2, x3, x4)
    top = min(y1, y2, y3, y4)
    right = max(x1, x2, x3, x4)
    bottom = max(y1, y2, y3, y4)

    original_crop = original_image_cv2[top:bottom, left:right]
    modified_crop = modified_image_cv2[top:bottom, left:right]

    if adjust_temp:
        temp_original = calcular_temperatura(original_crop)
        temp_modificada = calcular_temperatura(modified_crop)
        diferencia_temperatura = temp_original - temp_modificada
        modified_crop_adjusted = ajustar_temperatura(modified_crop, diferencia_temperatura)
    else:
        modified_crop_adjusted = modified_crop

    original_image_cv2[top:bottom, left:right] = modified_crop_adjusted
    combined_image = Image.fromarray(cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2RGB))

    return combined_image