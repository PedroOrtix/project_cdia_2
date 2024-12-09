from td_inpaint import inpaint
from inpaint_functions import parse_bounds
from ocr_utils import recalcular_cuadricula_rotada, convert_paddle_to_easyocr
from image_utils import rellenar_imagen_uniformemente, juntar_imagenes_vertical
from utils import separar_cadenas, mostrar_diccionario_ascii
from crop_compare import recortar_imagen
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

def simple_inpaint(image, bounds, word, slider_step=30, slider_guidance=2, slider_batch=6):
    """
    Realiza el proceso de inpainting en la imagen dada utilizando los límites y palabra especificados.
    
    Args:
        image (PIL.Image): La imagen en la que se realizará el inpainting.
        bounds (str): Los límites para el inpainting.
        word (str): La palabra para el inpainting.
        slider_step (int, opcional): El tamaño del paso para el slider. Por defecto 30.
        slider_guidance (float, opcional): El valor de guía para el slider. Por defecto 2.
        slider_batch (int, opcional): El tamaño del lote para el slider. Por defecto 6.
        
    Returns:
        tuple: La imagen con el inpainting realizado y las coordenadas.
    """
    global_dict = {}
    global_dict["stack"] = parse_bounds(bounds, word)
    # print(global_dict["stack"])   
    #image = "./hat.jpg"
    prompt = ""
    keywords = ""
    positive_prompt = ""
    radio = 8
    slider_natural= False
    return inpaint(image, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict)

def process_image(palabra, 
                    replace, 
                    bounds, 
                    img_array,
                    height=512,
                    width=512,
                    slider_step=30,
                    slider_guidance=2,
                    slider_batch=6,
                    show_plot=False,
                    save_intermediate_images=True):
    """
    Procesa una imagen identificando una palabra, reemplazándola por una nueva y devolviendo
    las imágenes modificadas. Opcionalmente, muestra las imágenes modificadas en una cuadrícula.

    Parámetros:
    ----------
    palabra : str
        La palabra a identificar y reemplazar en la imagen.
    replace : str
        La palabra que reemplazará a la palabra identificada.
    bounds : list
        Las coordenadas del cuadro delimitador de la palabra en la imagen original.
    img_array : np.ndarray
        El array de la imagen donde se encuentra la palabra.
    height : int, opcional
        La altura de la imagen redimensionada. Por defecto 512.
    width : int, opcional
        El ancho de la imagen redimensionada. Por defecto 512.
    slider_step : int, opcional
        El tamaño del paso para el slider. Por defecto 30.
    slider_guidance : float, opcional
        El valor de guía para el slider. Por defecto 2.
    slider_batch : int, opcional
        El tamaño del lote para el slider. Por defecto 6.
    show_plot : bool, opcional
        Si es True, muestra las imágenes modificadas en una cuadrícula. Por defecto False.
    save_intermediate_images : bool, opcional
        Si es True, guarda las imágenes intermedias durante el procesamiento. Por defecto True.

    Retorna:
    -------
    list de PIL.Image
        Una lista que contiene las imágenes modificadas después del reemplazo de la palabra.
    right_bounds : list
        Las coordenadas del cuadro delimitador de la palabra identificada en la imagen modificada.
    coordenadas_originales : list
        Las coordenadas originales del cuadro delimitador de la palabra en la imagen.
    """
    # Step 1: Resize and crop the image based on the bounding box and the word to be replaced
    img_resized, coordenadas_originales = recortar_imagen(bounds, palabra, img_array, alto=height, ancho=width)
    img_pil = Image.fromarray(img_resized).convert('RGB')
    # juntamos la misma imagen verticalmente para que sea cuadrada junto al papping en blanco
    # unicamente cuando el doble del alto sea menor que que 512px que lo que admite el modelo
    if 2*height <= 512:
        img_pil = juntar_imagenes_vertical(img_pil, img_pil)

    # dimesion de la imagen nueva con la adición vertical para futura restauración del padding
    img_pil = rellenar_imagen_uniformemente(img_pil, dimensiones_objetivo=(512, 512))
        
    # Save the cropped and resized image for reference (optional)
    if save_intermediate_images:
        img_pil.save("images/imagen_dni_recortada.jpg")

    # Inicializar el modelo OCR Paddle
    model = PaddleOCR(use_angle_cls=True, lang='es')  # 'es' para español
    # Step 2: Perform OCR on the resized and cropped image to identify word bounding boxes
    bounds_resized = model.ocr(np.array(img_pil))
    bounds_resized = convert_paddle_to_easyocr(bounds_resized)
    
    # mostramos otra vez las palabras detectadas, pero de la imagen recortada 512x512
    # reason: para que el usuario pueda ver las palabras detectadas y elegir la que desea reemplazar
    # el modelo ocr puede detectar en la nueva imagen recortada palabras que no estaban en la imagen original
    lista_elementos_detectados = []
    for bound in bounds_resized:
        lista_elementos_detectados.append(bound[1])
    
    dict_elems = separar_cadenas(lista_elementos_detectados)
    mostrar_diccionario_ascii(dict_elems)
    
    # ask the user to input the word to be replaced
    print("Por favor, introduzca la palabra a reemplazar: (por motivos de la demo)")
    palabra = input()

    # Step 3: Find the bounding box that matches the word to be replaced
    right_bounds = next(([bound] for bound in bounds_resized if bound[1] == palabra), None)
    if right_bounds is None:
        raise ValueError(f"No se encontró la palabra '{palabra}' en la imagen")

    # Step 4: Recalculate the bounding box if necessary (e.g., if the replacement word is longer)
    if palabra.isalpha() and len(palabra) < len(replace.strip()):
        right_bounds[0] = list(right_bounds[0])
        right_bounds[0][0] = recalcular_cuadricula_rotada(right_bounds[0][0], palabra, replace)

    # Step 5: Perform inpainting to replace the word in the image
    modified_images, composed_prompt = simple_inpaint(img_pil,
                                                    right_bounds,
                                                    [replace],
                                                    slider_step=slider_step,
                                                    slider_guidance=slider_guidance,
                                                    slider_batch=slider_batch)

    # Step 6: Optionally display the results in a grid format (3x2)
    if show_plot:
        fig, axs = plt.subplots(slider_batch//2, 2, figsize=(10, 10))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(modified_images[i])
            ax.axis('off')
            ax.set_title(f"Resultado {i+1}")

        plt.show()

    return modified_images, right_bounds, coordenadas_originales