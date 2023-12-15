import cv2
import numpy as np
import sys
import re

from pytesseract import pytesseract

TAMANHO_JANELA = 1280, 720

'''
Função para exibir imagem em uma janela, com tamanho fixo e título auto-explicativo sobre o que está
sendo exibido.
'''
def mostrar_imagem(img, titulo):
  cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(titulo, *TAMANHO_JANELA)
  cv2.imshow(titulo, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def enquadrar_imagem(img):
    # Encontrar coordenadas dos pixels não nulos
  pontos_nulos = cv2.findNonZero(img)

  # Encontrar retângulo delimitador em torno dos pixels não nulos
  x, y, w, h = cv2.boundingRect(pontos_nulos)

  # Recortar a região de interesse (ROI)
  roi = img[y:y+h, x:x+w]

  # Redimensionar a imagem recortada
  img = cv2.resize(roi, (TAMANHO_JANELA[0], TAMANHO_JANELA[1]), interpolation=cv2.INTER_CUBIC)

  return img

try:
  img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
except:
  print("Uso: python trabfinal.py <image>")
  sys.exit(1)

pytesseract.tesseract_cmd = r'tesseract'

# Equalização de histograma para melhorar contraste da imagem
img = cv2.equalizeHist(img)
mostrar_imagem(img, 'Equalizacao de histograma')

# Máscara a partir de aplicação de limiar mais suave, de modo a conectar os arredores da placa
_, bordas = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
mostrar_imagem(bordas, 'Limiarizacao para detectar os arredores da placa')

# Conversão em imagem binária aplicando um limiar
_, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
mostrar_imagem(img, 'Limizarizacao')

# Inversão de cores para que os números da placa fiquem em branco
img = cv2.bitwise_not(img)
bordas = cv2.bitwise_not(bordas)
mostrar_imagem(img, 'Inversao de cores')

# Extração de componentes conexo com coleta de estatísticas
num_componentes, rotulos, estatisticas, centroides \
  = cv2.connectedComponentsWithStats(bordas)

# Encontrar maior componente conexo da imagem exceto o fundo, que corresponde aos arredores da placa
maior_componente = np.argmax(estatisticas[1:, cv2.CC_STAT_AREA]) + 1

# Criar máscara para o maior componente conexo
mascara_maior_componente = (rotulos == maior_componente).astype(np.uint8) * 255
mostrar_imagem(mascara_maior_componente, 'Maior componente conexo')

# Remover o maior componente da imagem
img = cv2.subtract(img, mascara_maior_componente)
mostrar_imagem(img, 'Remocao do maior componente conexo')

kernel = np.ones((2, 2), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = 2)
mostrar_imagem(img, 'Fechamento')

# Aplicar abertura
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 5)
mostrar_imagem(img, 'Abertura')

# Enquadrar imagem
img = enquadrar_imagem(img)
mostrar_imagem(img, 'Imagem enquadrada')

# Suavizar imagem com filtro gaussiano
# img = cv2.GaussianBlur(img, (3, 3), 0)
# mostrar_imagem(img, 'Suavizacao com filtro gaussiano')

texto_detectado = pytesseract.image_to_string(img, lang='por')
print(texto_detectado)