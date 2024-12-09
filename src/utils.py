import re
from prettytable import PrettyTable

def separar_cadenas(lista):
    """
    Separa una lista de cadenas en dos categorías: alfabéticas y alfanuméricas.
    
    Args:
        lista (list): Lista de cadenas a procesar.
        
    Returns:
        dict: Diccionario con dos claves:
            - 'alpha': Lista de cadenas que solo contienen letras.
            - 'numeric': Lista de cadenas que contienen letras y/o números.
    """
    # Inicializar el diccionario
    resultado = {
        "alpha": [],
        "numeric": []
    }
    
    # Expresiones regulares
    regex_alpha = re.compile(r'^[A-Z]+$', re.IGNORECASE)  # Solo letras
    regex_numeric = re.compile(r'^[A-Z0-9]+$', re.IGNORECASE)  # Letras y/o números
    
    # Recorrer cada elemento de la lista
    for elemento in lista:
        if regex_alpha.match(elemento):
            resultado["alpha"].append(elemento)
        elif regex_numeric.match(elemento):
            resultado["numeric"].append(elemento)
    
    return resultado

def mostrar_diccionario_ascii(diccionario):
    """
    Muestra un diccionario en formato de tabla ASCII utilizando PrettyTable.
    
    Args:
        diccionario (dict): Diccionario a mostrar en formato tabla.
        
    Returns:
        None: Imprime la tabla en la consola.
    """
    # Crear una tabla
    tabla = PrettyTable()

    # Establecer los encabezados de las columnas
    tabla.field_names = ["Clave", "Valores"]

    # Iterar sobre el diccionario y agregar las filas a la tabla
    for clave, valores in diccionario.items():
        valores_formateados = "\n".join(valores)
        tabla.add_row([clave, valores_formateados])

    # Mostrar la tabla en formato ASCII
    print(tabla)