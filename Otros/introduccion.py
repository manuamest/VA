import cv2 as cv
img = cv.imread("images/images.jpeg")
imagen_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('imagen_gris.jpg', imagen_gris)
cv.imshow("Display window", imagen_gris)
k = cv.waitKey(0) # Wait for a keystroke in the window

