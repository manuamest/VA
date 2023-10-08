import numpy as np

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

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetro σ
    sigma = 1.5

    # Calcular el kernel Gaussiano
    kernel = gaussKernel1D(sigma)

    print("Kernel Gaussiano 1D:")
    print(kernel)
