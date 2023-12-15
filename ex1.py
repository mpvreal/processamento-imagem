'''
Aluno: Matheus Pires Vila Real    Matrícula: 202000560352
######### Exercício 1 #########
'''
import cv2 as cv
import numpy as np

from sys import argv

def amostragem(imagem, fator):
  altura, largura = imagem.shape[:2]
  nova_altura = altura // fator
  nova_largura = largura // fator
  
  nova_imagem = np.zeros((nova_altura, nova_largura, 3), dtype=np.uint8)

  for i in range(nova_altura):
    for j in range(nova_largura):
      nova_imagem[i, j] = imagem[i * fator, j * fator]
  
  return nova_imagem

try:
  caminho = argv[1]
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem> <fator>')
  exit()

imagem = cv.imread(caminho)
if imagem is None:
  print(f'Erro ao abrir a imagem em "{caminho}"')
  exit()

try:
  cv.namedWindow(f'{caminho}', cv.WINDOW_NORMAL)
  cv.resizeWindow(f'{caminho}', 600, 600)
  cv.imshow(f'{caminho}', amostragem(imagem, fator=int(argv[2])))
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem> <fator>')
  exit()

cv.waitKey(0)
cv.destroyAllWindows()