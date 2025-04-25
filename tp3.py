import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

# Crear carpeta "images" si no existe
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Descargar imagen de muestra (lena.jpg)
image_url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
image_path = os.path.join(output_dir, 'original.png')
if not os.path.exists(image_path):
    urllib.request.urlretrieve(image_url, image_path)

# 1. Cargar imagen en color (BGR)
image_bgr = cv2.imread(image_path)

# Convertir de BGR a RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Visualizar y guardar la imagen original en RGB
plt.imshow(image_rgb)
plt.title("Imagen Original (RGB)")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "original_rgb.png"))
plt.show()

# 2. Conversión a escala de grises
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap='gray')
plt.title("Imagen en Escala de Grises")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "gray.png"))
plt.show()

# 3. Reducción de niveles de gris
def reduce_gray_levels(image, levels):
    factor = 256 // levels
    reduced = (image // factor) * factor
    return reduced.astype(np.uint8)

levels_list = [2, 4, 8, 16, 32, 64, 128, 256]

for level in levels_list:
    reduced_img = reduce_gray_levels(image_gray, level)
    plt.imshow(reduced_img, cmap='gray')
    plt.title(f"{level} niveles de gris")
    plt.axis('off')
    filename = f"gray_{level}_levels.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# 4. Detección de bordes con Canny
edges = cv2.Canny(image_gray, 100, 200)
plt.imshow(edges, cmap='gray')
plt.title("Detección de Bordes - Canny")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "canny_edges.png"))
plt.show()
