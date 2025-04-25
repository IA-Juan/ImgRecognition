import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request


class ImageProcessor:
    def __init__(self, image_path=None, output_dir="images"):
        print(f"🛠️ Inicializando con image_path={image_path}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if image_path:
            self.image_path = image_path
            print(f"📷 Usando imagen proporcionada: {self.image_path}")
        else:
            # Descargar imagen por defecto (Lenna)
            self.image_path = os.path.join(self.output_dir, 'default.png')
            if not os.path.exists(self.image_path):
                url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
                urllib.request.urlretrieve(url, self.image_path)
                print("🌐 Imagen por defecto descargada.")

        self.image_bgr = cv2.imread(self.image_path)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def save_image(self, image, name, cmap=None):
        path = os.path.join(self.output_dir, name)
        if cmap:
            plt.imsave(path, image, cmap=cmap)
        else:
            plt.imsave(path, image)

    def reduce_gray_levels(self, image, levels):
        factor = 256 // levels
        reduced = (image // factor) * factor
        return reduced.astype(np.uint8)

    def process_all(self):
        """Muestra todo en un solo canvas: original, gris, reducciones y bordes"""
        levels_list = [2, 4, 8, 16, 32, 64, 128, 256]
        total_imgs = 2 + len(levels_list) + 1  # original + gray + reducciones + bordes

        cols = 4
        rows = (total_imgs + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(16, 12))
        axs = axs.flatten()

        idx = 0

        # Original RGB
        axs[idx].imshow(self.image_rgb)
        axs[idx].set_title("Original (RGB)")
        axs[idx].axis('off')
        self.save_image(self.image_rgb, "original_rgb.png")
        idx += 1

        # Escala de Grises
        axs[idx].imshow(self.image_gray, cmap='gray')
        axs[idx].set_title("Escala de Grises")
        axs[idx].axis('off')
        self.save_image(self.image_gray, "gray.png", cmap='gray')

        unique_levels = np.unique(self.image_gray)
        print(f"📊 La imagen en escala de grises tiene {len(unique_levels)} niveles de gris únicos.")
        idx += 1

        # Reducciones de niveles
        for level in levels_list:
            reduced = self.reduce_gray_levels(self.image_gray, level)
            axs[idx].imshow(reduced, cmap='gray')
            axs[idx].set_title(f"{level} niveles")
            axs[idx].axis('off')
            self.save_image(reduced, f"gray_{level}_levels.png", cmap='gray')
            idx += 1

        # Bordes - Canny
        edges = cv2.Canny(self.image_gray, 100, 200)
        axs[idx].imshow(edges, cmap='gray')
        axs[idx].set_title("Bordes (Canny)")
        axs[idx].axis('off')
        self.save_image(edges, "canny_edges.png", cmap='gray')

        # Quitar ejes vacíos si sobran
        for i in range(idx + 1, len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

# 🔧 Uso de la clase
if __name__ == "__main__":
    import sys

    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    processor = ImageProcessor(image_arg)
    processor.process_all()