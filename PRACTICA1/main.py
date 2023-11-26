import cv2
import numpy as np
import matplotlib

from adjustIntensity import adjustIntensity, run_adjust_intensity
from equalizeIntensity import equalizeIntensity, run_equalize_intensity
from filterImage import filterImage, run_filterImage
from gaussKernel1D import gaussKernel1D
from gaussianFilter import gaussianFilter, run_gaussianFilter
from medianFilter import medianFilter, run_medianFilter
from morph import run_morph
from hit_or_miss import hit_or_miss, run_hitormiss
from gradientImage import gradientImage, run_gradientImage
from LoG import LoG, run_LoG
from edgeCanny import edgeCanny, run_edgeCanny



if __name__ == "__main__":
    inImage = cv2.imread('fotos_manu/gradient_laplacianogaussiano.png', cv2.IMREAD_GRAYSCALE) / 255.0
    op = ['Erode', 'Dilate', 'Opening', 'Closing']
    op2 = ['Roberts', 'CentralDiff', 'Prewitt', 'Sobel']

    #run_adjust_intensity(inImage)
    #run_equalize_intensity(inImage)
    #run_filterImage(inImage)
    #run_gaussianFilter(inImage)
    #run_medianFilter(inImage)
    #run_morph(inImage, op[0])
    #run_hitormiss(inImage)
    #run_gradientImage(inImage, op2[1])
    #run_LoG(inImage)
    #run_edgeCanny(inImage)


