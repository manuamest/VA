import cv2 as cv
import numpy as np

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):

    # Verificar si el rango de entrada está vacío, en ese caso calcularlo
    if not inRange:
        inRange = [np.min(inImage), np.max(inImage)]
    
    # Aplicar la compresión o estiramiento lineal del histograma
    outImage = outRange[0] + (outRange[1] - outRange[0]) * (inImage - inRange[0]) / (inRange[1] - inRange[0])
    
    # Asegurarse de que los valores estén en el rango [0, 1]
    outImage = np.clip(outImage, 0, 1)
    
    return outImage

def equalizeIntensity(inImage, nBins=256):

    # Asegurarse de que la imagen está en el rango [0, 1]
    inImage = np.clip(inImage, 0, 1)

    # Crear un histograma acumulativo
    hist, _ = np.histogram(inImage * (nBins - 1), bins=nBins, range=(0, nBins - 1)) # Calculo del histograma
    histacum = hist.cumsum()                                                        # Histograma acumulativo
    histacum = histacum / histacum[-1]                                              # Normaliza entre 0 1

    # Ecualizar la imagen usando el histograma acumulativo
    outImage = histacum[np.round(inImage * (nBins - 1)).astype(int)]
    
    # Asegurarse de que la imagen de salida está en el rango [0, 1]
    outImage = np.clip(outImage, 0, 1)

    return outImage

def filterImage(inImage, kernel):
    # Obtener las dimensiones de la imagen de entrada y el kernel
    rows, cols = inImage.shape
    kRows, kCols = kernel.shape

    # Calcular el desplazamiento necesario para centrar el kernel
    dRow = kRows // 2
    dCol = kCols // 2

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage, dtype=np.float32)

    # Convolución
    for i in range(dRow, rows - dRow):
        for j in range(dCol, cols - dCol):
            # Extraer la región de interés de la imagen de entrada
            roi = inImage[i - dRow:i + dRow + 1, j - dCol:j + dCol + 1]

            # Aplicar la convolución entre el kernel y la región de interés
            conv_result = np.sum(roi * kernel)

            # Asignar el resultado a la posición correspondiente en la imagen de salida
            outImage[i, j] = conv_result

    return outImage

def gaussKernel1D(sigma):
    # Calcular N a partir de σ
    N = int(2 * np.ceil(3 * sigma) + 1)

    # Calcular el centro del kernel
    center = N // 2

    # Inicializar el kernel como un vector de ceros
    kernel = np.zeros(N)

    # Calcular el valor del kernel Gaussiano
    for i in range(N):
        x = i - center
        kernel[i] = np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    # Normalizar el kernel para que la suma sea igual a 1
    kernel /= np.sum(kernel)

    return kernel

def gaussianFilter(inImage, sigma):
    # Calcular los kernels unidimensionales
    kernel1D = gaussKernel1D(sigma)

    # Aplicar la convolución horizontal (1D)
    filtered_horizontal = np.apply_along_axis(lambda x: np.convolve(x, kernel1D, mode='same'), axis=1, arr=inImage)

    # Aplicar la convolución vertical (1D) al resultado de la convolución horizontal
    filtered_image = np.apply_along_axis(lambda x: np.convolve(x, kernel1D, mode='same'), axis=0, arr=filtered_horizontal)

    return filtered_image

def medianFilter(inImage, filterSize):
    # Obtener las dimensiones de la imagen de entrada
    rows, cols = inImage.shape

    # Mitad del tamaño del filtro (radio)
    radius = filterSize // 2 + 1

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Iterar sobre la imagen
    for i in range(rows):
        for j in range(cols):
            # Definir los límites de la ventana
            row_min = max(0, i - radius)
            row_max = min(rows, i + radius + 1)
            col_min = max(0, j - radius)
            col_max = min(cols, j + radius + 1)

            # Extraer la región de interés (ventana)
            window = inImage[row_min:row_max, col_min:col_max]

            # Calcular la mediana de la ventana y asignarla al píxel de salida
            outImage[i, j] = np.median(window)

    return outImage

# TENER EN CUENTA QUE LAS IMAGENES TIENEN QUE ESTAR EN BINARIO _, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)


def erode(inImage, SE, center=None):
    # Si el centro no se proporciona, se calcula
    if center is None:
        center = [SE.shape[0]//2, SE.shape[1]//2]

    # Crear una imagen de salida del mismo tamaño que la imagen de entrada
    outImage = np.zeros_like(inImage)

    # Recorrer cada píxel de la imagen
    for i in range(center[0], inImage.shape[0]-center[0]):
        for j in range(center[1], inImage.shape[1]-center[1]):
            # Aplicar el elemento estructurante al píxel y su vecindario
            neighborhood = inImage[i-center[0]:i+(SE.shape[0]-center[0]), j-center[1]:j+(SE.shape[1]-center[1])]
            # Si todos los píxeles bajo el elemento estructurante son 1, el píxel se mantiene
            if (neighborhood[SE==1] == 1).all():
                outImage[i, j] = 1

    return outImage

def dilate(inImage, SE, center=None):
    # Si el centro no se proporciona, se calcula
    if center is None:
        center = [SE.shape[0]//2, SE.shape[1]//2]

    # Crear una imagen de salida del mismo tamaño que la imagen de entrada
    outImage = np.zeros_like(inImage)

    # Recorrer cada píxel de la imagen
    for i in range(center[0], inImage.shape[0]-center[0]):
        for j in range(center[1], inImage.shape[1]-center[1]):
            # Aplicar el elemento estructurante al píxel y su vecindario
            neighborhood = inImage[i-center[0]:i+(SE.shape[0]-center[0]), j-center[1]:j+(SE.shape[1]-center[1])]
            # Si cualquier píxel bajo el elemento estructurante es 1, el píxel se mantiene
            if (neighborhood[SE==1] == 1).any():
                outImage[i, j] = 1

    return outImage

def opening(inImage, SE, center=None):
    return dilate(erode(inImage, SE, center), SE, center)

def closing(inImage, SE, center=None):
    return erode(dilate(inImage, SE, center), SE, center)

#           HIT OR MISS PENDIENTE

