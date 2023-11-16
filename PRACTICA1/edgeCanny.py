import cv2
import numpy as np

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

def gradientImage(inImage, operator):

    # Seleccion de operador
    if operator == 'Roberts':
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=np.float32)
    elif operator == 'CentralDiff':
        kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    elif operator == 'Prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    elif operator == 'Sobel':
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    else:
        return "Error: operador no reconocido"
    
    # Calculo gx y gy
    gx = filterImage(inImage, kernel_x)
    gy = filterImage(inImage, kernel_y)

    return gx, gy

def edgeCanny(inImage, sigma, tlow, thigh):
    # Paso 1: Aplicar filtro Gaussiano para suavizar la imagen
    smoothed_image = gaussianFilter(inImage, sigma)
    cv2.imshow('Smoothed Image', smoothed_image)


    # Paso 2: Calcular gradientes en las direcciones x e y con el operador Sobel
    [gradient_x, gradient_y] = gradientImage(smoothed_image, 'Sobel')

    # Paso 3: Calcular la magnitud del gradiente y la dirección
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    # Paso 4: Aplicar la supresión no máxima (non-maximum suppression)
    gradient_direction = np.round(gradient_direction / 45) * 45
    gradient_direction[gradient_direction < 0] += 180

    suppressed_image = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            direction = gradient_direction[i, j]
            if direction == 0 and (magnitude[i, j] >= magnitude[i, j - 1]) and (magnitude[i, j] >= magnitude[i, j + 1]):
                suppressed_image[i, j] = magnitude[i, j]
            elif direction == 45 and (magnitude[i, j] >= magnitude[i - 1, j + 1]) and (magnitude[i, j] >= magnitude[i + 1, j - 1]):
                suppressed_image[i, j] = magnitude[i, j]
            elif direction == 90 and (magnitude[i, j] >= magnitude[i - 1, j]) and (magnitude[i, j] >= magnitude[i + 1, j]):
                suppressed_image[i, j] = magnitude[i, j]
            elif direction == 135 and (magnitude[i, j] >= magnitude[i - 1, j - 1]) and (magnitude[i, j] >= magnitude[i + 1, j + 1]):
                suppressed_image[i, j] = magnitude[i, j]

    cv2.imshow('Suppressed Image', suppressed_image)

    # Paso 5: Aplicar el umbral de histéresis
    edges = np.zeros_like(suppressed_image)
    
    strong_edges = (suppressed_image >= thigh)
    weak_edges = (suppressed_image >= tlow) & (suppressed_image <= thigh)

    # Etapa de seguimiento de bordes débiles conectados a bordes fuertes
    for i in range(1, suppressed_image.shape[0] - 1):
        for j in range(1, suppressed_image.shape[1] - 1):
            if weak_edges[i, j]:
                # Verificar vecinos para conexión a bordes fuertes
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    edges[i, j] = 1

    cv2.imshow('Edges Image', edges)

    # Mantener bordes fuertes
    edges[strong_edges] = 1
    
    return edges.astype(np.float32)

# Cargar una imagen de ejemplo
original_image = cv2.imread("imgp1/chica4k.png", cv2.IMREAD_GRAYSCALE) / 255.0

# Parámetros de Canny
sigma = 1.4
tlow = 30/255
thigh = 100/255

# Aplicar el detector de bordes de Canny
result_image = edgeCanny(original_image, sigma, tlow, thigh)

# Mostrar la imagen original y el resultado
cv2.imshow('Original Image', original_image)
cv2.imshow('Canny Edge Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
