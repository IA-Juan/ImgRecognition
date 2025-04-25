import os
import sys
import cv2
import urllib.request
from urllib.parse import urlparse
from datetime import datetime
import matplotlib.pyplot as plt
import math

class ImageProcessor:
    """
    Clase para procesar imágenes utilizando OpenCV y Matplotlib.
    Realiza operaciones como conversión a escala de grises, reducción de niveles de gris, 
    detección de bordes y visualización de las imágenes resultantes.
    """

    def __init__(self, image_path=None, output_dir="images"):
        """
        Inicializa el objeto ImageProcessor.
        
        :param image_path: Ruta local o URL de la imagen a procesar. Si no se proporciona, se descarga una imagen por defecto.
        :param output_dir: Directorio donde se guardarán las imágenes procesadas. Por defecto, 'images'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Si se pasa una ruta de imagen o URL
        if image_path:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                # Si es una URL, descarga la imagen
                parsed_url = urlparse(image_path)
                self.image_name = os.path.basename(parsed_url.path)
                self.image_path = os.path.join(self.output_dir, self.image_name)
                if not os.path.exists(self.image_path):
                    print(f"🌐 Descargando imagen desde URL: {image_path}")
                    urllib.request.urlretrieve(image_path, self.image_path)
                    print(f"✅ Imagen descargada como {self.image_path}")
                else:
                    print(f"📁 Usando imagen previamente descargada: {self.image_path}")
            elif os.path.exists(image_path):
                # Si es una ruta local válida
                self.image_path = os.path.abspath(image_path)
                self.image_name = os.path.basename(self.image_path)
                print(f"📷 Usando imagen local: {self.image_path}")
            else:
                # Si el archivo no existe, se maneja el error
                print(f"❌ El archivo {image_path} no existe.")
                user_input = input("¿Desea continuar con la imagen predeterminada? (s/n): ").strip().lower()
                if user_input == 's':
                    self.image_name = 'Lenna_test_image.png'
                    self.image_path = os.path.join(self.output_dir, self.image_name)
                    if not os.path.exists(self.image_path):
                        url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
                        print("🌐 Descargando imagen por defecto...")
                        urllib.request.urlretrieve(url, self.image_path)
                        print("✅ Imagen por defecto descargada.")
                    else:
                        print(f"📁 Usando imagen por defecto: {self.image_path}")
                else:
                    print("👋 El proceso se detendrá.")
                    sys.exit()  # Sale del programa si el usuario no quiere continuar con la imagen predeterminada
        else:
            # Si no se pasa ningún parámetro, se descarga una imagen predeterminada
            self.image_name = 'Lenna_test_image.png'
            self.image_path = os.path.join(self.output_dir, self.image_name)
            if not os.path.exists(self.image_path):
                url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
                print("🌐 Descargando imagen por defecto...")
                urllib.request.urlretrieve(url, self.image_path)
                print("✅ Imagen por defecto descargada.")

        # Prefijo para los nombres de los archivos (sin extensión)
        self.image_prefix = os.path.splitext(self.image_name)[0]
        print(f"📎 Prefijo para guardado: {self.image_prefix}")

        # Cargar la imagen usando OpenCV
        self.image_bgr = cv2.imread(self.image_path)
        if self.image_bgr is None:
            raise FileNotFoundError(f"❌ No se pudo cargar la imagen desde: {self.image_path}")

        # Convertir la imagen de BGR (OpenCV) a RGB (para Matplotlib)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        # Convertir a escala de grises
        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def get_image_depth(self):
        """
        Muestra la profundidad de la imagen (en bits por canal).

        :return: La profundidad de la imagen (dtype de la matriz de imagen).
        """
        depth = self.image_bgr.dtype
        print(f"La profundidad de la imagen es: {depth}")
        return depth

    def reduce_gray_levels(self, gray_image, levels):
        """
        Reduce la cantidad de niveles de gris de una imagen en escala de grises.

        :param gray_image: Imagen en escala de grises.
        :param levels: Número de niveles de gris a mantener.
        :return: Imagen con los niveles de gris reducidos.
        """
        if levels > 256:
            print(f"⚠️ Nivel {levels} excede los 256 posibles en imágenes de 8 bits. Se limitará a 256.")
            levels = 256
        factor = 256 // levels
        reduced = (gray_image // factor) * factor
        return reduced

    def save_image(self, image, suffix):
        """
        Guarda una imagen procesada con un sufijo específico.

        :param image: Imagen a guardar.
        :param suffix: Sufijo para el nombre del archivo.
        """
        filename = os.path.join(self.output_dir, f"{self.image_prefix}_{suffix}.png")
        plt.imsave(filename, image, cmap='gray' if len(image.shape) == 2 else None)
        print(f"💾 Imagen guardada: {filename}")

    def show_all(self):
        """
        Muestra todas las imágenes procesadas y las guarda en el directorio de salida.
        Incluye la imagen original, la versión en escala de grises, las versiones con reducción de niveles de gris 
        y los bordes detectados por Canny.
        """
        images = []
        titles = []

        # Agregar la imagen RGB original
        images.append(self.image_rgb)
        titles.append(f"{self.image_prefix}_original_rgb.png")
        self.save_image(self.image_rgb, "original_rgb")

        # Agregar la imagen en escala de grises
        images.append(self.image_gray)
        titles.append(f"{self.image_prefix}_gray.png")
        self.save_image(self.image_gray, "gray")

        # Agregar imágenes con diferentes niveles de gris
        for level in [2, 4, 8, 16, 32, 64, 128, 256]:
            reduced = self.reduce_gray_levels(self.image_gray, level)
            images.append(reduced)
            titles.append(f"{self.image_prefix}_gray_{level}_levels.png")
            self.save_image(reduced, f"gray_{level}_levels")

        # Detección de bordes con Canny
        canny = cv2.Canny(self.image_gray, 100, 200)
        images.append(canny)
        titles.append(f"{self.image_prefix}_canny_edges.png")
        self.save_image(canny, "canny_edges")

        # Calcular número de filas y columnas para el mosaico
        total = len(images)
        cols = 4
        rows = math.ceil(total / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
        fig.suptitle(f"🔍 Visualización de Imágenes Procesadas: {self.image_prefix}", fontsize=16)
        axes = axes.flatten()

        # Mostrar cada imagen en el mosaico
        for idx, (img, title) in enumerate(zip(images, titles)):
            ax = axes[idx]
            cmap = 'gray' if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        # Apagar los ejes de las subgráficas vacías
        for j in range(len(images), len(axes)):
            axes[j].axis('off')

        # Agregar pie de imagen con la fecha actual y texto adicional
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"Procesado el {fecha_actual} - Grupo Vintage"
        fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, color='gray')

        # Ajustar el layout y guardar el canvas completo
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.08)

        # Guardar el canvas completo
        canvas_filename = os.path.join(self.output_dir, f"{self.image_prefix}_canvas.png")
        fig.savefig(canvas_filename)
        print(f"🖼️ Mosaico guardado como: {canvas_filename}")

        # Cambiar el título de la ventana y mostrar la figura
        fig.canvas.manager.set_window_title(f"Mosaico - {self.image_prefix}")
        plt.show()


if __name__ == '__main__':
    # Obtener la ruta de la imagen desde los argumentos de la línea de comandos
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    processor = ImageProcessor(image_path)  # Crear el objeto ImageProcessor
    processor.get_image_depth()  # Mostrar la profundidad de la imagen
    processor.show_all()  # Mostrar todas las imágenes procesadas
