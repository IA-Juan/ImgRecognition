import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request


class ImageProcessor:
    def __init__(self, image_url, output_dir="images"):
        self.image_url = image_url
        self.output_dir = output_dir
        self.image_path = os.path.join(self.output_dir, 'original.png')

        os.makedirs(self.output_dir, exist_ok=True)
        self.download_image()
        self.image_bgr = cv2.imread(self.image_path)
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def download_image(self):
        if not os.path.exists(self.image_path):
            urllib.request.urlretrieve(self.image_url, self.image_path)

    def save_image(self, image, name, cmap=None):
        path = os.path.join(self.output_dir, name)
        if cmap:
            plt.imsave(path, image, cmap=cmap)
        else:
            plt.imsave(path, image)
    
    def show_original(self):
        """Muestra y guarda la imagen original en RGB."""
        self.save_image(self.image_rgb, "original_rgb.png")

        plt.imshow(self.image_rgb)
        plt.title("Imagen Original (RGB)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_gray(self):
        """Muestra y guarda la imagen en escala de grises, e indica cuántos niveles de gris únicos tiene."""
        self.save_image(self.image_gray, "gray.png", cmap='gray')

        plt.imshow(self.image_gray, cmap='gray')
        plt.title("Escala de Grises")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        unique_levels = np.unique(self.image_gray)
        print(f"La imagen en escala de grises tiene {len(unique_levels)} niveles de gris únicos.")

    def reduce_gray_levels(self, levels):
        factor = 256 // levels
        reduced = (self.image_gray // factor) * factor
        return reduced.astype(np.uint8)

    def show_reduced_grays(self, levels_list=None):
        if levels_list is None:
            levels_list = [2, 4, 8, 16, 32, 64, 128, 256]

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))

        for idx, level in enumerate(levels_list):
            reduced_img = self.reduce_gray_levels(level)
            row, col = divmod(idx, 4)
            axs[row, col].imshow(reduced_img, cmap='gray')
            axs[row, col].set_title(f"{level} niveles")
            axs[row, col].axis('off')
            self.save_image(reduced_img, f"gray_{level}_levels.png", cmap='gray')

        plt.tight_layout()
        plt.show()

    def show_canny_edges(self, threshold1=100, threshold2=200):
        edges = cv2.Canny(self.image_gray, threshold1, threshold2)
        self.save_image(edges, "canny_edges.png", cmap='gray')

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(edges, cmap='gray')
        ax.set_title("Detección de Bordes (Canny)")
        ax.axis('off')
        plt.tight_layout()
        plt.show()


# 🔧 Uso de la clase
if __name__ == "__main__":
    image_url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    processor = ImageProcessor(image_url)

    processor.show_original()
    processor.show_gray()
    processor.show_reduced_grays()
    processor.show_canny_edges()
