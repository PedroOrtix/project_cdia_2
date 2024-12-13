import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Función para calcular y graficar histogramas superpuestos
def plot_superimposed_histograms(imagen_original_path, imagen_modificada_path, output_path):
    """
    Genera un gráfico de líneas superpuestas con los histogramas de dos imágenes.
    
    Interpretación:
    - Las líneas muestran la distribución de brillos en cada imagen
    - Picos altos indican muchos píxeles con ese nivel de brillo
    - Desplazamientos horizontales indican cambios en el brillo general
    - Cambios en la forma indican modificaciones en el contraste
    """
    # Cargar imágenes y convertirlas a escala de grises
    imagen_original = np.array(Image.open(imagen_original_path).convert("L"))
    imagen_modificada = np.array(Image.open(imagen_modificada_path).convert("L"))

    # Calcular histogramas (256 bins para valores 0-255)
    hist_original, bins_original = np.histogram(imagen_original.ravel(), bins=256, range=(0, 256), density=True)
    hist_modificada, bins_modificada = np.histogram(imagen_modificada.ravel(), bins=256, range=(0, 256), density=True)

    # Calcular los centros de los bins
    bins_original_center = (bins_original[:-1] + bins_original[1:]) / 2
    bins_modificada_center = (bins_modificada[:-1] + bins_modificada[1:]) / 2

    # Crear el gráfico con Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    sns.lineplot(x=bins_original_center, y=hist_original, 
                label="Imagen Original - Distribución de intensidades de píxeles antes del procesamiento", 
                color="blue")
    sns.lineplot(x=bins_modificada_center, y=hist_modificada, 
                label="Imagen Modificada - Distribución de intensidades de píxeles después del procesamiento", 
                color="orange")

    # Añadir título y etiquetas
    plt.title("Comparación de Distribución de Intensidades de Píxeles", fontsize=14)
    plt.xlabel("Intensidad de píxel (0=negro, 255=blanco)", fontsize=12)
    plt.ylabel("Frecuencia Normalizada", fontsize=12)
    plt.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.0, 1.0))

    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pixel_difference(imagen_original_path, imagen_modificada_path, output_path):
    """
    Genera un mapa de calor que muestra las diferencias absolutas entre dos imágenes.
    
    Interpretación:
    - Zonas rojas/brillantes indican grandes diferencias entre las imágenes
    - Zonas azules/oscuras indican pocas o ninguna diferencia
    - Útil para identificar áreas específicas donde el procesamiento tuvo mayor impacto
    - La intensidad del color indica la magnitud del cambio
    """
    # Cargar imágenes y convertir a escala de grises
    img1 = np.array(Image.open(imagen_original_path).convert("L"))
    img2 = np.array(Image.open(imagen_modificada_path).convert("L"))
    
    # Calcular diferencia absoluta
    diferencia = np.abs(img1.astype(float) - img2.astype(float))
    
    # Crear figura
    plt.figure(figsize=(12, 6))
    
    # Subplot para las imágenes originales
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Imagen Modificada')
    plt.axis('off')
    
    # Subplot para el mapa de calor
    plt.subplot(1, 3, 3)
    mapa = plt.imshow(diferencia, cmap='hot')
    plt.colorbar(mapa, label='Diferencia absoluta')
    plt.title('Mapa de Diferencias')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_color_histograms(imagen_original_path, imagen_modificada_path, output_path):
    """
    Genera histogramas RGB superpuestos para dos imágenes.
    
    Interpretación:
    - Cada gráfico representa la distribución de un canal de color (R,G,B)
    - Cambios en la forma indican modificaciones en la saturación del color
    - Desplazamientos horizontales indican cambios en la intensidad del color
    - Útil para detectar alteraciones específicas en cada canal de color
    """
    # Cargar imágenes en modo RGB
    img1 = np.array(Image.open(imagen_original_path).convert("RGB"))
    img2 = np.array(Image.open(imagen_modificada_path).convert("RGB"))
    
    canales = ['Rojo', 'Verde', 'Azul']
    colores = ['red', 'green', 'blue']
    
    plt.figure(figsize=(15, 5))
    
    for idx, (canal, color) in enumerate(zip(range(3), colores)):
        plt.subplot(1, 3, idx + 1)
        
        # Histograma para imagen original
        hist1, bins = np.histogram(img1[:,:,canal].ravel(), bins=256, range=(0, 256), density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(centers, hist1, color=color, linestyle='-', label='Original', alpha=0.7)
        
        # Histograma para imagen modificada
        hist2, _ = np.histogram(img2[:,:,canal].ravel(), bins=256, range=(0, 256), density=True)
        plt.plot(centers, hist2, color=color, linestyle='--', label='Modificada', alpha=0.7)
        
        plt.title(f'Canal {canales[idx]}')
        plt.xlabel('Intensidad')
        plt.ylabel('Frecuencia')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_image_overlay(imagen_original_path, imagen_modificada_path, output_path, alpha=0.5):
    """
    Superpone dos imágenes con transparencia ajustable.
    
    Interpretación:
    - Permite visualizar directamente las diferencias espaciales entre imágenes
    - Áreas donde las imágenes difieren mostrarán un efecto "fantasma"
    - Áreas sin cambios aparecerán nítidas
    - Útil para identificar desplazamientos o cambios estructurales
    """
    # Cargar imágenes
    img1 = np.array(Image.open(imagen_original_path))
    img2 = np.array(Image.open(imagen_modificada_path))
    
    # Asegurar que ambas imágenes tengan el mismo tamaño
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Crear superposición
    overlay = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title('Imagen Modificada')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Superposición')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_fft_difference(imagen_original_path, imagen_modificada_path, output_path):
    """
    Visualiza las diferencias en el dominio de la frecuencia entre dos imágenes.
    
    Interpretación:
    - El centro representa las frecuencias bajas (cambios graduales)
    - Los bordes representan frecuencias altas (detalles finos)
    - Diferencias brillantes en el centro indican cambios en el contraste global
    - Diferencias en los bordes indican cambios en los detalles finos o ruido
    - Patrones simétricos sugieren cambios sistemáticos en la imagen
    """
    # Cargar imágenes y convertir a escala de grises
    img1 = np.array(Image.open(imagen_original_path).convert("L"))
    img2 = np.array(Image.open(imagen_modificada_path).convert("L"))
    
    # Calcular FFT
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    
    # Calcular el espectro de magnitud y centrarlo
    magnitude1 = np.abs(np.fft.fftshift(fft1))
    magnitude2 = np.abs(np.fft.fftshift(fft2))
    
    # Calcular diferencia
    diff_magnitude = np.abs(magnitude1 - magnitude2)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.log1p(magnitude1), cmap='viridis')
    plt.title('FFT Original')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log1p(magnitude2), cmap='viridis')
    plt.title('FFT Modificada')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.log1p(diff_magnitude), cmap='viridis')
    plt.title('Diferencia FFT')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_temperature_comparison(imagen1_path, imagen2_path, output_path):
    """
    Genera una visualización comparativa de las temperaturas/intensidades entre dos imágenes diferentes.
    
    Interpretación:
    - Los mapas de calor muestran la distribución de temperaturas en cada imagen
    - El gráfico de dispersión muestra la correlación de temperaturas entre imágenes
    - Los histogramas muestran la distribución de temperaturas en cada imagen
    - Las estadísticas permiten comparar numéricamente ambas imágenes
    """
    # Cargar imágenes y convertir a escala de grises
    img1 = np.array(Image.open(imagen1_path).convert("L")).astype(float)
    img2 = np.array(Image.open(imagen2_path).convert("L")).astype(float)
    
    # Crear un mapa de color para temperaturas
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
    cmap_temp = LinearSegmentedColormap.from_list('temp', colors, N=256)
    
    # Crear la figura con subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Mapa de calor imagen 1
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(img1, cmap=cmap_temp)
    plt.colorbar(im1, ax=ax1, label='Temperatura/Intensidad')
    ax1.set_title('Imagen 1')
    ax1.axis('off')
    
    # Mapa de calor imagen 2
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(img2, cmap=cmap_temp)
    plt.colorbar(im2, ax=ax2, label='Temperatura/Intensidad')
    ax2.set_title('Imagen 2')
    ax2.axis('off')
    
    # Histogramas superpuestos
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.hist(img1.ravel(), bins=50, alpha=0.5, label='Imagen 1', color='blue')
    ax_hist.hist(img2.ravel(), bins=50, alpha=0.5, label='Imagen 2', color='red')
    ax_hist.set_title('Distribución de Temperaturas')
    ax_hist.set_xlabel('Temperatura/Intensidad')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.legend()
    
    # Gráfico de dispersión
    ax_scatter = fig.add_subplot(gs[1, 0:2])
    # Modificar el tamaño de la muestra para que sea dinámico
    sample_size = min(5000, img1.size)  # Tomar el mínimo entre 5000 y el tamaño de la imagen
    indices = np.random.choice(img1.size, sample_size, replace=False)
    ax_scatter.scatter(img1.ravel()[indices], img2.ravel()[indices], 
                      alpha=0.1, c='blue', s=1)
    ax_scatter.set_xlabel('Temperatura Imagen 1')
    ax_scatter.set_ylabel('Temperatura Imagen 2')
    ax_scatter.set_title('Correlación de Temperaturas')
    
    # Añadir línea de referencia y = x
    min_val = min(img1.min(), img2.min())
    max_val = max(img1.max(), img2.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    # Estadísticas comparativas
    stats_text = f"""Estadísticas Comparativas:
    
    Imagen 1:
    Media: {np.mean(img1):.2f}
    Mediana: {np.median(img1):.2f}
    Desv. Est.: {np.std(img1):.2f}
    
    Imagen 2:
    Media: {np.mean(img2):.2f}
    Mediana: {np.median(img2):.2f}
    Desv. Est.: {np.std(img2):.2f}
    
    Correlación: {np.corrcoef(img1.ravel(), img2.ravel())[0,1]:.3f}"""
    
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_stats.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

