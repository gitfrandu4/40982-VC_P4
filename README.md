# Práctica 4. Reconocimiento de matrículas

Este repositorio contiene la **Práctica 4** donde se utilizan técnicas de procesamiento de video para la detección y seguimiento de vehículos y personas, aplicando reconocimiento de caracteres a las matrículas visibles. Para ello, hacemos uso de modelos YOLO para detectar objetos y OCRs para el reconocimiento de texto.

# Índice

- [Práctica 4. Reconocimiento de matrículas](#práctica-4-reconocimiento-de-matrículas)
- [Índice](#índice)
  - [Librerías utilizadas](#librerías-utilizadas)
  - [Autores](#autores)
  - [Tareas](#tareas)
    - [Tarea 1](#tarea-1)
  - [Referencias y bibliografía](#referencias-y-bibliografía)

## Librerías utilizadas

[![YOLO](https://img.shields.io/badge/YOLO-v11-00FFFF?style=for-the-badge&logo=yolo)](https://ultralytics.com/yolo)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-5A5A5A?style=for-the-badge&logo=tesseract)](https://github.com/tesseract-ocr/tesseract)
[![easyOCR](https://img.shields.io/badge/EasyOCR-FFD700?style=for-the-badge&logo=easyocr)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)


## Autores

Este proyecto fue desarrollado por:

- [![GitHub](https://img.shields.io/badge/GitHub-Francisco%20Javier%20L%C3%B3pez%E2%80%93Dufour%20Morales-yellow?style=flat-square&logo=github)](https://github.com/gitfrandu4)
- [![GitHub](https://img.shields.io/badge/GitHub-Marcos%20V%C3%A1zquez%20Tasc%C3%B3n-purple?style=flat-square&logo=github)](https://github.com/DerKom)

## Tareas

### Tarea 1

#### Instalar Real-ESRGAN

1. Primero, navega a la carpeta de tu proyecto y ejecuta los siguientes comandos:

   ```bash
   # Instalar Git (si no lo tienes ya instalado)
   conda install git

   # Clonar el repositorio de Real-ESRGAN
   git clone https://github.com/xinntao/Real-ESRGAN

   # Navegar dentro del directorio del proyecto
   cd Real-ESRGAN

   # Instalar dependencias
   pip install basicsr
   pip install facexlib
   pip install gfpgan
   pip install -r requirements.txt

   # Configurar el entorno de desarrollo
   python setup.py develop
   ```

2. Descargar el modelo preentrenado:
   
   Descarga el archivo del modelo desde el siguiente enlace y colócalo en la carpeta raíz del proyecto:  
   [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
   [RealESRGAN_x4plus_anime_6B.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

##### Corrección de Error Conocido

Para resolver un error que puede ocurrir, edita el archivo `degradations.py` en tu entorno.

1. Localiza el archivo en:
   ```plaintext
   C:\Users\<tu_usuario>\anaconda3\envs\VC_P4\Lib\site-packages\basicsr\data\degradations.py
   ```

2. Busca la línea:

   ```python
   from torchvision.transforms.functional_tensor import rgb_to_grayscale
   ```

3. Reemplázala por:

   ```python
   from torchvision.transforms.functional import rgb_to_grayscale
   ```

Esto debería corregir el problema y completar el proceso de instalación.

---

## Referencias y bibliografía

- YOLO Documentation: [ultralytics.com/yolo](https://docs.ultralytics.com/)
- PyTorch Documentation: [pytorch.org](https://pytorch.org/docs/)
- Tesseract Documentation: [github.com/tesseract-ocr](https://github.com/tesseract-ocr/tesseract)
- EasyOCR Documentation: [github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- OpenCV Documentation: [docs.opencv.org](https://docs.opencv.org/)
- CUDA Documentation: [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)

