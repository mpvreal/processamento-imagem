'''
Aluno: Matheus Pires Vila Real    Matrícula: 202000560352
######### Exercício 2 #########
'''
import cv2 as cv
import numpy as np

from sys import argv

def quantizacao(imagem, n_cores: int):
  if n_cores < 1 or n_cores > 256:
    raise ValueError('O número de cores deve estar entre 1 e 256')

  altura, largura = imagem.shape[:2]
  imagem = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
  nova_imagem = np.zeros((altura, largura, 3), dtype=np.uint8)

  for x in range(altura):
    for y in range(largura):
      nova_imagem[x, y] = np.round(imagem[x, y] * n_cores / 255.0) * (255.0 / n_cores)

  return nova_imagem

try:
  caminho = argv[1]
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem> <nº-de-cores>')
  exit()

imagem = cv.imread(caminho)
if imagem is None:
  print(f'Erro ao abrir a imagem em "{caminho}"')
  exit()

try:
  cv.namedWindow(f'{caminho}', cv.WINDOW_NORMAL)
  cv.resizeWindow(f'{caminho}', 600, 600)
  cv.imshow(f'{caminho}', quantizacao(imagem, int(argv[2])))
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem> <nº-de-cores>')
  exit()

cv.waitKey(0)
cv.destroyAllWindows()