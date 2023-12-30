import cv2 as cv
import numpy as np
from campo2 import campo

def is_collision(rect1, rect2):
    # Verifica si hay colisión entre dos rectángulos
    # rect1 y rect2 son tuplas de la forma (x, y, w, h)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Verificar si algún lado de un rectángulo está más allá del lado correspondiente del otro
    if x1 > x2 + w2 or x2 > x1 + w1:
        return False
    if y1 > y2 + h2 or y2 > y1 + h1:
        return False
    return True

def line_to_rect(line, line_thickness=1):
    # Convierte una línea en un rectángulo delgado
    x1, y1, x2, y2 = line

    if abs(x2 - x1) > abs(y2 - y1):  # Línea más horizontal
        return (min(x1, x2), min(y1, y2) - line_thickness//2, abs(x2 - x1), line_thickness)
    else:  # Línea más vertical
        return (min(x1, x2) - line_thickness//2, min(y1, y2), line_thickness, abs(y2 - y1))

def is_line_collision(line1, line2, line_thickness=1):
    # Convierte las líneas en rectángulos delgados
    rect1 = line_to_rect(line1, line_thickness)
    rect2 = line_to_rect(line2, line_thickness)

    # Usa la función existente para verificar la colisión entre rectángulos
    return is_collision(rect1, rect2)

def longitud_linea(linea):
    x1, y1, x2, y2 = linea
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def detectar_lineas_siega(inImage, mask):
    
    campo_image = inImage.copy()

    # Closing operation to remove noise and holes in the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

    mask_refined = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)

    # Convertir la imagen del campo a escala de grises
    gray_image = cv.cvtColor(campo_image, cv.COLOR_BGR2GRAY)
    
    # Aplicar un filtro Gaussiano para reducir el ruido
    blurred = cv.GaussianBlur(gray_image, (21, 21), 0)
    equalized = cv.equalizeHist(blurred)
    
    # Aplicar un filtro de detección de bordes verticales en ambas direcciones
    sobelx = cv.Sobel(equalized, cv.CV_64F, 1, 0, ksize=7)
    sobelx_neg = -sobelx  # Negativo del filtro Sobel

    # Umbralizar las imágenes para resaltar los bordes verticales fuertes
    _, thresholded_pos = cv.threshold(np.abs(sobelx), 200, 255, cv.THRESH_BINARY)
    _, thresholded_neg = cv.threshold(np.abs(sobelx_neg), 200, 255, cv.THRESH_BINARY)
    
    # Combinar ambos thresholded images
    thresholded_combined = cv.bitwise_or(thresholded_pos, thresholded_neg,  mask=mask_refined)
    
    # Aplicar la máscara del campo para evitar detectar contornos de jugadores
    thresholded_combined = cv.bitwise_and(thresholded_combined, thresholded_combined, mask=mask_refined)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))
    closing = cv.morphologyEx(thresholded_combined, cv.MORPH_ERODE, kernel)
    #cv.imshow("Closing", closing)

    # Usar la Transformada de Hough para encontrar líneas
    lines = cv.HoughLinesP(closing.astype(np.uint8), 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=200)

    drawn_lines = []  # Lista para mantener las líneas ya dibujadas

    if lines is not None:
        for line in lines:
            current_line = line[0]
            current_length = longitud_linea(current_line)
            collision_found = False

            for i, existing_line in enumerate(drawn_lines):
                if is_line_collision(current_line, existing_line, line_thickness=3):
                    existing_length = longitud_linea(existing_line)
                    if current_length > existing_length:
                        # Reemplazar la línea existente con la nueva línea
                        drawn_lines[i] = current_line
                        collision_found = True
                        break

            if not collision_found:
                # Si no hay colisión, agregar la línea a la lista
                drawn_lines.append(current_line)

    # Dibujar las líneas finales
    for line in drawn_lines:
        x1, y1, x2, y2 = line
        cv.line(campo_image, (x1, y1), (x2, y2), (0, 255, 255), 3)

    return campo_image

def run_perspectiva(inImage):
    imagen, mask = campo(inImage)
    resultado = detectar_lineas_siega(imagen, mask)
    cv.imshow("Original", inImage)
    cv.imshow("Resultado perspectiva", resultado)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":

    inImage = cv.imread("Material_Futbol/174.jpg")

    imagen, mask = campo(inImage)
    resultado = detectar_lineas_siega(imagen, mask)

    #cv.imshow('Original', inImage)
    cv.imshow("Perspectiva", resultado)

    cv.waitKey(0)
    cv.destroyAllWindows()