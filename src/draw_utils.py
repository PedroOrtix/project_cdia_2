from PIL import ImageDraw, ImageFont

def fit_text_in_box(draw, text, box, font_path='arial.ttf'):
    """
    Ajusta el tamaño del texto para que quepa dentro de una caja delimitadora.
    
    Args:
        draw (PIL.ImageDraw.Draw): Objeto Draw de PIL para dibujar.
        text (str): Texto a ajustar.
        box (tuple): Tupla de 4 puntos que definen la caja delimitadora.
        font_path (str, opcional): Ruta al archivo de fuente. Por defecto 'arial.ttf'.
        
    Returns:
        PIL.ImageFont.FreeTypeFont: Objeto fuente con el tamaño ajustado.
    """
    p0, p1, p2, p3 = box
    box_width = max(p1[0] - p0[0], p2[0] - p3[0])
    box_height = max(p3[1] - p0[1], p2[1] - p1[1])
    
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    
    while text_width > box_width or text_height > box_height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
    
    return font

def draw_boxes(image, bounds, color='yellow', width=2, fill_color=None, replace=None):
    """
    Dibuja cajas delimitadoras en una imagen.
    
    Args:
        image (PIL.Image): Imagen sobre la que dibujar.
        bounds (list): Lista de límites (bounds) para dibujar.
        color (str, opcional): Color de las líneas. Por defecto 'yellow'.
        width (int, opcional): Ancho de las líneas. Por defecto 2.
        fill_color (str, opcional): Color de relleno de las cajas. Por defecto None.
        replace (list, opcional): Lista de textos para reemplazar. Por defecto None.
        
    Returns:
        PIL.Image: Imagen con las cajas dibujadas.
    """
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        box = bound[0]
        p0, p1, p2, p3 = box
        if fill_color:
            draw.polygon([*p0, *p1, *p2, *p3], fill=fill_color)
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=(0,255,0), width=width)
    return image

def draw_mask(image, bounds, color='yellow', width=2, fill_color=None, replace=None):
    """
    Dibuja máscaras y texto en una imagen.
    
    Args:
        image (PIL.Image): Imagen sobre la que dibujar.
        bounds (list): Lista de límites (bounds) para dibujar.
        color (str, opcional): Color de las líneas. Por defecto 'yellow'.
        width (int, opcional): Ancho de las líneas. Por defecto 2.
        fill_color (str, opcional): Color de relleno de las máscaras. Por defecto None.
        replace (list, opcional): Lista de textos para reemplazar. Por defecto None.
        
    Returns:
        PIL.Image: Imagen con las máscaras y texto dibujados.
    """
    draw = ImageDraw.Draw(image)
    for i, bound in enumerate(bounds):
        box = bound[0]
        text = replace[i] if replace else bound[1]
        p0, p1, p2, p3 = box
        
        if fill_color:
            draw.polygon([*p0, *p1, *p2, *p3], fill=fill_color)
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=(193,193,193), width=width)
        
        font = fit_text_in_box(draw, text, box)
        text_width, text_height = draw.textsize(text, font=font)
        text_x = p0[0] + (p1[0] - p0[0] - text_width) / 2
        text_y = p0[1] + (p3[1] - p0[1] - text_height) / 2
        draw.text((text_x, text_y), text, fill=(0,0,0), font=font)
    
    return image 