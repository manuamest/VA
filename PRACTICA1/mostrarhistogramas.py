import cv2
import matplotlib.pyplot as plt
import numpy as np

def mostrarhistogramas(original, modificada):
    nBins = 256

    # Histograma de la imagen original
    histOriginal, _ = np.histogram(original.flatten(), nBins)
    cumHistOriginal = np.cumsum(histOriginal)

    # Histograma de la imagen modificada
    histModificada, _ = np.histogram(modificada.flatten(), nBins)
    cumHistModificada = np.cumsum(histModificada)

    # Configurar la disposición de los subgráficos
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Mostrar la imagen original
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title('Imagen Original')

    # Mostrar el histograma de la imagen original
    axs[0, 1].plot(histOriginal, color='red', label='Histograma')
    axs[0, 1].set_title('Histograma Original')
    axs[0, 1].set_ylim(0, 5000)
    axs[0, 1].legend()

    # Mostrar la imagen modificada
    axs[1, 0].imshow(modificada, cmap='gray')
    axs[1, 0].set_title('Imagen Modificada')

    # Mostrar el histograma de la imagen modificada
    axs[1, 1].plot(histModificada, color='blue', label='Histograma')
    axs[1, 1].set_title('Histograma Modificado')
    axs[1, 1].set_ylim(0, 5000)
    plt.tight_layout()

    # Crear una nueva figura para los histogramas acumulados
    fig, ax = plt.subplots(figsize=(8, 4))

    # Histograma acumulado
    ax.plot(cumHistOriginal, color='red', label='Original')
    ax.plot(cumHistModificada, color='blue', label='Modificado')
    ax.set_title('Comparacion Histogramas Acumulados')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    original = cv2.imread('fotos_manu/adjust_filter.png', cv2.IMREAD_GRAYSCALE) / 255.0
    modificada = cv2.imread('resultados/adjustIntensity.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
    mostrarhistogramas(original, modificada)