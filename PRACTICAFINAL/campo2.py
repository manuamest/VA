import cv2 as cv
import numpy as np

def campo(imagen):
    inImage = imagen.copy()
    # Convert image to HSV color space
    hsvImage = cv.cvtColor(inImage, cv.COLOR_BGR2HSV)

    # Fixed range for green grass (might need tweaking)
    lower_region = np.array([25, 40, 40])
    upper_region = np.array([70, 255, 255])

    # Create a mask for the green grass
    mask = cv.inRange(hsvImage, lower_region, upper_region)

    # Closing operation to remove noise and holes in the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Find contours on the mask
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # If contours were found, proceed to find the largest one
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        epsilon = 0.015 * cv.arcLength(largest_contour, True)
        approx = cv.approxPolyDP(largest_contour, epsilon, True)

        # Draw the approximated polygon on a blank mask
        blank = np.zeros(inImage.shape[:2], dtype='uint8')
        cv.drawContours(blank, [approx], -1, 255, thickness=cv.FILLED)

        # Bitwise-AND with the original image to get the field area
        resultado = cv.bitwise_and(inImage, inImage, mask=blank)
        return resultado, closing

    # Return the original image if no grass is detected
    return inImage

def run_campo(imagen):

    resultado, _ = campo(imagen)

    cv.imshow('Original', imagen)
    cv.imshow("Campo", resultado)
    cv.imwrite("SOLOCAMPO.jpg", resultado)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":

    imagen = cv.imread('Material_Futbol/6.jpg')


    resultado, _ = campo(imagen)

    cv.imshow('Original', imagen)
    cv.imshow("Campo", resultado)
    cv.imwrite("SOLOCAMPO.jpg", resultado)

    cv.waitKey(0)
    cv.destroyAllWindows()