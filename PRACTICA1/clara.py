def adjustIntensity(inImage, inRange = [], outRange = [0, 1]):

    # Comprobar se inRange está baleiro ou non. En caso de estalo establecer min e max da imaxe de entrada
    if len(inRange) == 0:
        inRange = [np.min(inImage), np.max(inImage)]
    
    if len(inRange) != 2 or len(outRange) != 2:
        raise ValueError("inRange e outRange teñen que ser de lonxitude 2")

    inMin, inMax = inRange
    outMin, outMax = outRange

    inRangeLen = inMax - inMin
    outRangeLen = outMax -outMin

    # Aplicar a modificación do rango de intensidade a cada elemento da matriz
    newIntensity = lambda x: outMin + (outRangeLen * (x - inMin)) / inRangeLen
    outImage = np.vectorize(newIntensity)(inImage)

    return outImage