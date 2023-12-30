import cv2 as cv

def white_pixels_image(image):
    # Pasa imagen a escala de grises
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Elemento estrcututal (CROSS 15X15)
    filterSize =(15, 15)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, filterSize)
    # Tratamiento de Top-Hat x2
    tophat_img = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    tophat_img2 = cv.morphologyEx(tophat_img, cv.MORPH_TOPHAT, kernel)
    # De escala de grises a RGB
    bgr = cv.cvtColor(tophat_img2, cv.COLOR_GRAY2BGR)
    image3 = cv.resize(bgr, (960, 540))
    #cv.imshow("white_pixels_image", image3)
    return bgr

