'''
Aluno: Matheus Pires Vila Real    Matrícula: 202000560352
######### Exercício 3 #########
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

def pixels_adjacentes(imagem, i, j):
  altura, largura = imagem.shape[:2]
  adjacencia = []

  # Definir os offsets para os vizinhos em 8 direções
  offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1), (1, 0), (1, 1)]

  for offset in offsets:
    p_i = i + offset[0]
    p_j = j + offset[1]

    # Verificar se o novo pixel está dentro dos limites da imagem
    if 0 <= p_i < altura and 0 <= p_j < largura:
      if np.array_equal(imagem[p_i, p_j], imagem[i, j]):
        adjacencia.append((p_i, p_j))

  return adjacencia

def componentes_conexos(imagem):
  greyscale = quantizacao(imagem, 1)
  nova_imagem = np.zeros_like(greyscale)
  altura, largura = greyscale.shape[:2]
  fila = []
  cor = np.array([np.random.randint(1, 255), np.random.randint(1, 255), np.random.randint(1, 255)])

  for i in range(altura):
    for j in range(largura):
      if np.array_equal(nova_imagem[i, j], [0, 0, 0]):
        nova_imagem[i, j] = cor
        fila.append((i, j))

      while len(fila) > 0:
        p = fila.pop(0)

        for a in pixels_adjacentes(greyscale, p[0], p[1]):
          if np.array_equal(nova_imagem[a[0], a[1]], [0, 0, 0]):
            nova_imagem[a[0], a[1]] = cor
            fila.append((a[0], a[1]))

    cor = np.array([np.random.randint(1, 255), np.random.randint(1, 255), np.random.randint(1, 255)])

  return nova_imagem

try:
  caminho = argv[1]
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem>')
  exit()

imagem = cv.imread(caminho)
if imagem is None:
  print(f'Erro ao abrir a imagem em "{caminho}"')
  exit()

try:
  cv.namedWindow(f'{caminho}', cv.WINDOW_NORMAL)
  cv.resizeWindow(f'{caminho}', 600, 600)
  cv.imshow(f'{caminho}', componentes_conexos(imagem))
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem>')
  exit()

cv.waitKey(0)
cv.destroyAllWindows()