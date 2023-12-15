import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

try:
  path = argv[1]
  print(f'Imagem carregada: {path}')
except:
  print("Usage: python3 passabaixa.py <path to image>")
  exit()

img = cv.imread(path)

# Imagem suavizada usando filtro gaussiano
suavizada = cv.GaussianBlur(img, (5, 5), 0)

cv.imshow('Imagem Suavizada', suavizada)
cv.waitKey(0)
cv.destroyAllWindows()

# Imagem em passa-alta
passa_alta = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
passa_alta = cv.Laplacian(passa_alta, cv.CV_8U)

cv.imshow('Imagem Passa-Alta', passa_alta)
cv.waitKey(0)
cv.destroyAllWindows()
