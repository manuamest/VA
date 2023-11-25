import cv2
import numpy as np

from adjustIntensity import adjustIntensity, run_adjust_intensity
from equalizeIntensity import equalizeIntensity, run_equalize_intensity
from filterImage import filterImage, run_filterImage
from gaussKernel1D import gaussKernel1D
from gaussianFilter import gaussianFilter, run_gaussianFilter
from medianFilter import medianFilter, run_medianFilter
from morph import run_morph

if __name__ == "__main__":
    inImage = cv2.imread('resultados/morph.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
    op = ['Erode', 'Dilate', 'Opening', 'Closing']

    #run_adjust_intensity(inImage)
    #run_equalize_intensity(inImage)
    #run_filterImage(inImage)
    #run_gaussianFilter(inImage)        #Revisar traspuesta
    #run_medianFilter(inImage)
    #run_morph(inImage, op[0])          #Revisar extendImageDuplicate
    
