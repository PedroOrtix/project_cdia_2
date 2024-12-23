import cv2
import numpy as np

def calcular_temperatura(imagen):
    """
    Calcula la temperatura de color promedio de una imagen.
    
    Args:
        imagen (numpy.ndarray): Imagen en formato BGR.
        
    Returns:
        float: Valor promedio de temperatura de color.
    """
    imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    promedio_temperatura = np.mean(imagen_lab[:,:,2]) - np.mean(imagen_lab[:,:,0])
    return promedio_temperatura

def ajustar_temperatura(imagen, ajuste):
    """
    Ajusta la temperatura de color de una imagen.
    
    Args:
        imagen (numpy.ndarray): Imagen en formato BGR.
        ajuste (float): Valor de ajuste de temperatura.
        
    Returns:
        numpy.ndarray: Imagen con la temperatura ajustada.
    """
    imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    imagen_lab[:,:,2] = cv2.add(imagen_lab[:,:,2], int(ajuste))
    imagen_ajustada = cv2.cvtColor(imagen_lab, cv2.COLOR_LAB2BGR)
    imagen_ajustada = np.clip(imagen_ajustada, 0, 255).astype(np.uint8)
    return imagen_ajustada 