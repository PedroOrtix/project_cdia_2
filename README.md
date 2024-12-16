# 🔄 Detección y Reemplazo de Texto en Imágenes mediante Modelos de Difusión: 
## Un Análisis de TextDiffuser-2

🎯 Este repositorio contiene una implementación para la detección y reemplazo de texto en imágenes utilizando técnicas de OCR y text inpainting.

## 💻 Requisitos del Sistema

- 🐍 Python 3.10+
- 🎮 CUDA compatible GPU (recomendado)
- 🐧 Linux (probado en Fedora 41)

## 🚀 Instalación

1. 📥 Clonar el repositorio:
```bash
git clone https://github.com/PedroOrtix/project_cdia_2.git
cd project_cdia_2
```

2. 🌐 Crear un entorno virtual (recomendado):
```bash
conda create -n project_cdia_2 python=3.11
conda activate project_cdia_2  # En Linux/Mac
```

3. 📦 Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
.
├── 📚 articles/                      # Artículos y papers de referencia
├── 🖼️ images/                        # Imágenes de ejemplo y resultados
├── 📄 images_dni/                    # Imágenes de DNI para pruebas
├── 📓 notebooks/
│   ├── Florence_SAM.ipynb           # Notebook para análisis con Florence y SAM
│   ├── notebook_prueba.ipynb        # Notebook de pruebas
│   └── notebook_text_inpaint.ipynb  # Notebook principal con ejemplos
├── 📊 results/                       # Resultados de los experimentos
├── 🔧 src/
│   ├── crop_compare.py             # Utilidades de recorte y comparación
│   ├── draw_utils.py               # Utilidades de dibujo
│   ├── image_utils.py              # Utilidades de procesamiento de imágenes
│   ├── inpaint_functions.py        # Funciones principales de inpainting
│   ├── ocr_utils.py               # Utilidades de OCR
│   ├── plots.py                   # Funciones de visualización
│   ├── run.py                     # Script principal de ejecución
│   ├── td_inpaint.py             # Implementación de Text Detection Inpainting
│   ├── temperature_utils.py       # Utilidades de análisis de temperatura
│   └── utils.py                   # Utilidades generales
├── 📝 paper.tex                     # Documento LaTeX del paper
├── 📄 paper.pdf                     # PDF del paper
├── ⚖️ LICENSE                        # Licencia del proyecto
├── 📖 README.md                     # Este archivo
└── 📋 requirements.txt              # Dependencias del proyecto
```

## ⚠️ Notas Importantes

- 🎮 El modelo requiere una GPU con CUDA para un rendimiento óptimo
- 🖼️ Las imágenes de entrada deben estar en formato común (jpg, png)
- 📝 El texto detectado debe ser legible y estar en un contraste adecuado

## 🔧 Solución de Problemas

1. 🚨 Error de CUDA:
   - ✅ Verificar la instalación de CUDA
   - ✅ Confirmar compatibilidad de versiones