# Conjunto de datos MNIST

Los datos son imágenes de números del 0 al 9 en escala de grises. Cada imagen de MNIST tiene \(28 \times 28 = 784\) píxeles.

Las imágenes se almacenan en un archivo con la terminación `.idx3-ubyte`. Este tipo se usa para almacenar matrices de datos, especialmente imágenes en escala de grisis. 

Las etiquetas de las imágenes se almacenan en un archivo `.idx1-ubyte`. Un tipo parecido al anterior, pero específico para almacenar las etiquetas.

Estructura de los archivos:

- Número mágico (4 bytes): identifica el tipo de archivo; para imágenes suele ser 2051.
- Número de imágenes (4 bytes).
- Filas (4 bytes): normalmente 28.
- Columnas (4 bytes): normalmente 28.
- Datos de píxeles: valores entre 0 y 255 que representan la intensidad de cada píxel.

Contiene una cabecera de 8 bytes:
- Número mágico (4 bytes): identifica el tipo de archivo; para etiquetas suele ser 2051.
- Número de etiquetas.
- Bytes restantes: una etiqueta por imagen, con valores de 0 a 9.
- Datos de píxeles: valores entre 0 y 255 que representan la intensidad de cada píxel.