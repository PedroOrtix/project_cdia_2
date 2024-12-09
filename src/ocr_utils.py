import numpy as np

def convert_paddle_to_easyocr(paddle_result):
    easyocr_result = []
    for block in paddle_result[0]:
        bbox = block[0]
        text = block[1][0]
        confidence = block[1][1]
        easyocr_result.append((bbox, text, confidence))
    return easyocr_result

def recalcular_cuadricula_rotada(cuadricula, palabra_original, palabra_objetivo):
    longitud_original = len(palabra_original)
    longitud_maxima_objetivo = len(palabra_objetivo) 
    
    expansion_proporcion = longitud_maxima_objetivo / longitud_original
    
    p1 = np.array(cuadricula[0])
    p2 = np.array(cuadricula[1])
    p3 = np.array(cuadricula[2])
    p4 = np.array(cuadricula[3])
    
    vector_lado_superior = p2 - p1
    vector_lado_inferior = p3 - p4
    
    nuevo_vector_lado_superior = vector_lado_superior * expansion_proporcion
    nuevo_vector_lado_inferior = vector_lado_inferior * expansion_proporcion
    
    p2_nuevo = p1 + nuevo_vector_lado_superior
    p3_nuevo = p4 + nuevo_vector_lado_inferior
    
    nueva_cuadricula = [p1.tolist(), p2_nuevo.tolist(), p3_nuevo.tolist(), p4.tolist()]
    
    return nueva_cuadricula