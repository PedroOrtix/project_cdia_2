import torch
import numpy as np
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler,UNet2DConditionModel
from tqdm import tqdm
from PIL import Image, ImageDraw
import string

def add_tokens(tokenizer, text_encoder):
    """
    Añade tokens adicionales al tokenizador y redimensiona el codificador de texto.
    Incluye tokens de coordenadas y caracteres.
    
    Args:
        tokenizer (CLIPTokenizer): Tokenizador a modificar.
        text_encoder (CLIPTextModel): Codificador de texto a redimensionar.
    """
    #### additional tokens are introduced, including coordinate tokens and character tokens
    
    alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
    
    for i in range(520):
        tokenizer.add_tokens(['l' + str(i) ]) # left
        tokenizer.add_tokens(['t' + str(i) ]) # top
        tokenizer.add_tokens(['r' + str(i) ]) # width
        tokenizer.add_tokens(['b' + str(i) ]) # height    
    for c in alphabet:
        tokenizer.add_tokens([f'[{c}]']) 
        
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    print(f'Tokenizer: Added {520*4+95} tokens')




def format_prompt(draw, prompt, dict_stack):
    """
    Formatea un prompt añadiendo información de posición y texto para el proceso de inpainting.
    
    Args:
        draw (PIL.ImageDraw.Draw): Objeto para dibujar en la imagen.
        prompt (str): Prompt base.
        dict_stack (list): Lista de tuplas (posición, texto) para el inpainting.
        
    Returns:
        str: Prompt formateado con la información de posición y texto.
    """
    user_prompt = prompt + ' <|endoftext|><|startoftext|>'

    for items in dict_stack:
        position, text = items

        x0, y0, x1, y1, x2, y2, x3, y3 = position
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=1)
        x0 = x0 // 4
        y0 = y0 // 4
        x1 = x1 // 4
        y1 = y1 // 4
        x2 = x2 // 4
        y2 = y2 // 4
        x3 = x3 // 4
        y3 = y3 // 4
        xmin = min(x0, x1, x2, x3)
        ymin = min(y0, y1, y2, y3)
        xmax = max(x0, x1, x2, x3)
        ymax = max(y0, y1, y2, y3)
        text_str = ' '.join([f'[{c}]' for c in list(text)])
        user_prompt += f' l{xmin} t{ymin} r{xmax} b{ymax} {text_str} <|endoftext|>'

        print('prompt ', user_prompt)
    return user_prompt
        
def to_tensor(image):
    """
    Convierte una imagen a tensor de PyTorch.
    
    Args:
        image (PIL.Image o numpy.ndarray): Imagen a convertir.
        
    Returns:
        torch.Tensor: Tensor normalizado de la imagen.
        
    Raises:
        TypeError: Si la imagen no es del tipo esperado.
    """
    if isinstance(image, Image.Image):  
        image = np.array(image)
    elif not isinstance(image, np.ndarray):  
        raise TypeError("Error")

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image)

    return tensor

def parse_bounds(bounds, wordlist):
    """
    Parsea los límites y palabras para el formato requerido por el inpainting.
    
    Args:
        bounds (list): Lista de límites (bounds) de las palabras.
        wordlist (list): Lista de palabras a procesar.
        
    Returns:
        list: Lista formateada de límites y palabras.
    """
    wordlist += [''] * (len(bounds) - len(wordlist))
    new_bounds = []
    for i in range(len(bounds)):
        positions = []
        pos_list = bounds[i][0]
        for number_list in pos_list:
            positions.extend(number_list)
        new_bounds.append([positions, wordlist[i]])
    print(new_bounds)
    return new_bounds