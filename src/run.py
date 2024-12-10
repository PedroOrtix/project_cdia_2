from src.td_inpaint import inpaint
from src.inpaint_functions import parse_bounds
from src.ocr_utils import recalcular_cuadricula_rotada, convert_paddle_to_easyocr
from src.image_utils import (
    rellenar_imagen_uniformemente, 
    juntar_imagenes_vertical,
    juntar_imagenes_horizontal,
    recortar_imagen_uniformemente
)
from src.utils import separar_cadenas, mostrar_diccionario_ascii
from src.crop_compare import recortar_imagen, comparar_imagenes, reemplazar_parte_imagen
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
        Si es 256, la imagen se duplicará verticalmente para mantener el contexto.
    width : int, opcional
        El ancho de la imagen redimensionada. Por defecto 512.
        Si es 256, la imagen se duplicará horizontalmente para mantener el contexto.
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
    # Step 1: Resize and crop the image
    img_resized, coordenadas_originales = recortar_imagen(bounds, palabra, img_array, alto=height, ancho=width)
    img_pil = Image.fromarray(img_resized).convert('RGB')
    
    # Si height es 256, juntar verticalmente para mantener el contexto
    if height == 256:
        img_pil = juntar_imagenes_vertical(img_pil, img_pil)
    
    # Si width es 256, juntar horizontalmente
    if width == 256:
        img_pil = juntar_imagenes_horizontal(img_pil, img_pil)
    
    # Rellenar con padding si es necesario
    img_pil = rellenar_imagen_uniformemente(img_pil, dimensiones_objetivo=(512, 512))
        
    # Save the cropped and resized image for reference
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

    return modified_images, right_bounds, coordenadas_originales, img_pil

def process_document_image(
    input_image_path: str,
    output_dir: str = "images",
    steps: int = 30,
    guidance_scale: float = 2.0,
    batch_size: int = 6,
    height: int = 512,
    width: int = 512,
    save_all_versions: bool = True,
    show_comparison: bool = False,
    lang: str = 'es'
) -> list:
    """
    Procesa una imagen de documento para reemplazar texto específico de manera interactiva,
    siguiendo el pipeline del notebook.

    Args:
        input_image_path: Ruta de la imagen de entrada
        output_dir: Directorio donde guardar las imágenes resultantes
        steps: Número de pasos para el proceso de inpainting
        guidance_scale: Escala de guía para el proceso de inpainting
        batch_size: Número de variaciones a generar
        save_all_versions: Si es True, guarda todas las versiones generadas
        show_comparison: Si es True, muestra una comparación visual
        lang: Idioma para el OCR ('es' para español)
        height: Altura de la imagen redimensionada. Por defecto 512.
           Si es 256, la imagen se duplicará verticalmente.
        width: Ancho de la imagen redimensionada. Por defecto 512.
          Si es 256, la imagen se duplicará horizontalmente.

    Returns:
        list: Lista de rutas de todas las imágenes procesadas
    """
    import os
    from PIL import Image
    import numpy as np
    from paddleocr import PaddleOCR
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Primera detección de texto en la imagen original
    print("\n=== Primera Detección de Texto en la Imagen Original ===")
    print("Analizando imagen...")
    img = Image.open(input_image_path)
    img_array = np.array(img)
    
    model = PaddleOCR(use_angle_cls=True, lang=lang)
    result = model.ocr(img_array)
    bounds = convert_paddle_to_easyocr(result)
    
    # Mostrar elementos detectados de manera estética
    lista_elementos_detectados = [bound[1] for bound in bounds]
    dict_elems = separar_cadenas(lista_elementos_detectados)
    print("\n=== Elementos Detectados en Imagen Original ===")
    mostrar_diccionario_ascii(dict_elems)
    
    # 2. Solicitar primera entrada del usuario
    print("\n=== Selección de Texto a Reemplazar ===")
    word_to_replace = input("Por favor, introduzca el texto que desea reemplazar: ")
    replacement_word = input("Por favor, introduzca el nuevo texto: ")
    
    modified_images, right_bounds, coordenadas_originales, img_pil = process_image(
        palabra=word_to_replace,
        replace=replacement_word,
        bounds=bounds,
        img_array=img_array,
        height=height,
        width=width,
        slider_step=steps,
        slider_guidance=guidance_scale,
        slider_batch=batch_size,
        show_plot=show_comparison,
        save_intermediate_images=save_all_versions
    )
    
    # El resto del procesamiento permanece igual...
    processed_images = []
    for i, modified_image in enumerate(modified_images):
        # estaba img_pil, ahora ya la tenemos en el return de process_image
        coodinates = np.array(right_bounds[0][0], dtype=int).flatten()
        
        img_recortada_mod = reemplazar_parte_imagen(
            img_pil,
            modified_image,
            coodinates,
            adjust_temp=False
        )
        
        img_recortada_mod, _ = recortar_imagen_uniformemente(img_recortada_mod)
        # Recortar según las dimensiones originales
        if height == 256:
            img_recortada_mod = img_recortada_mod.crop(
                (0, 0, img_recortada_mod.width, img_recortada_mod.height//2)
            )
        if width == 256:
            img_recortada_mod = img_recortada_mod.crop(
                (0, 0, img_recortada_mod.width//2, img_recortada_mod.height)
            )
        
        img_array_copy = img_array.copy()[:, :, :3]
        x_min, y_min, x_max, y_max = coordenadas_originales
        img_array_copy[y_min:y_max, x_min:x_max] = np.array(img_recortada_mod)[:, :, :]
        
        final_path = os.path.join(output_dir, f"resultado_final_version_{i+1}.jpg")
        Image.fromarray(img_array_copy).save(final_path)
        processed_images.append(final_path)
        
        if show_comparison:
            comparar_imagenes(
                img_pil,
                modified_image,
                coodinates,
                save_path=os.path.join(output_dir, f"comparison_version_{i+1}"),
                adjust_temp=False
            )
    
    print("\n=== Proceso Completado ===")
    print(f"Se han generado {len(processed_images)} versiones")
    print(f"Las imágenes se han guardado en: {output_dir}")
    
    return processed_images