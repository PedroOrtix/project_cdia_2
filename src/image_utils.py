from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

def display(image):
    """
    Muestra una imagen utilizando matplotlib.
    
    Args:
        image (PIL.Image o numpy.ndarray): Imagen a mostrar.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def stitch(images):
    """
    Une horizontalmente una lista de imágenes.
    
    Args:
        images (list): Lista de imágenes PIL.Image a unir.
        
    Returns:
        PIL.Image: Imagen resultante de la unión horizontal.
    """
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    stitched = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        stitched.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return stitched

def juntar_imagenes_vertical(imagen_upper, imagen_lower):
    """
    Une verticalmente dos imágenes.
    
    Args:
        imagen_upper (PIL.Image): Imagen superior.
        imagen_lower (PIL.Image): Imagen inferior.
        
    Returns:
        PIL.Image: Imagen resultante de la unión vertical.
    """
    ancho_upper, alto_upper = imagen_upper.size
    ancho_lower, alto_lower = imagen_lower.size
    ancho_combined = max(ancho_upper, ancho_lower)
    alto_combined = alto_upper + alto_lower
    imagen_combined = Image.new('RGB', (ancho_combined, alto_combined))
    imagen_combined.paste(imagen_upper, (0, 0))
    imagen_combined.paste(imagen_lower, (0, alto_upper))
    return imagen_combined

def rellenar_imagen_uniformemente(imagen_pil, dimensiones_objetivo, color_relleno=(255, 255, 255)):
    """
    Rellena una imagen con márgenes uniformes hasta alcanzar las dimensiones objetivo.
    
    Args:
        imagen_pil (PIL.Image): Imagen a rellenar.
        dimensiones_objetivo (tuple): Tupla (ancho, alto) con las dimensiones deseadas.
        color_relleno (tuple, opcional): Color RGB para el relleno. Por defecto blanco (255, 255, 255).
        
    Returns:
        PIL.Image: Imagen rellenada con márgenes uniformes.
    """
    ancho_original, alto_original = imagen_pil.size
    ancho_objetivo, alto_objetivo = dimensiones_objetivo
    margen_izquierdo = (ancho_objetivo - ancho_original) // 2
    margen_superior = (alto_objetivo - alto_original) // 2
    margen_derecho = ancho_objetivo - ancho_original - margen_izquierdo
    margen_inferior = alto_objetivo - alto_original - margen_superior
    imagen_rellena = ImageOps.expand(imagen_pil, 
                                   border=(margen_izquierdo, margen_superior, margen_derecho, margen_inferior), 
                                   fill=color_relleno)
    return imagen_rellena

def recortar_imagen_uniformemente(imagen_pil, color_relleno=(255, 255, 255)):
    """
    Recorta los márgenes de una imagen de un color específico.
    
    Args:
        imagen_pil (PIL.Image): Imagen a recortar.
        color_relleno (tuple, opcional): Color RGB de los márgenes a recortar. Por defecto blanco (255, 255, 255).
        
    Returns:
        tuple: Tupla conteniendo:
            - PIL.Image: Imagen recortada.
            - tuple: Coordenadas del recorte (x_min, y_min, x_max, y_max).
    """
    imagen_np = np.array(imagen_pil)
    mask = np.any(imagen_np != color_relleno, axis=-1)
    if mask.any():
        y_min, y_max = np.where(mask)[0][[0, -1]]
        x_min, x_max = np.where(mask)[1][[0, -1]]
        imagen_recortada = imagen_pil.crop((x_min, y_min, x_max + 1, y_max + 1))
        return imagen_recortada, (x_min, y_min, x_max + 1, y_max + 1)
    return imagen_pil, (0, 0, imagen_pil.width, imagen_pil.height) 