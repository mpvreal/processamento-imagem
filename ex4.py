'''
Aluno: Matheus Pires Vila Real    Matrícula: 202000560352
######### Exercício 4 #########
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

try:
  caminho = argv[1]
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem>')
  exit()

imagem = cv.imread(caminho, cv.IMREAD_GRAYSCALE)

if imagem is None:
  print(f'Erro ao abrir a imagem em "{caminho}"')
  exit()

histograma, _ = np.histogram(imagem.flatten(), 256, [0, 256])
probabilidades = histograma / float(imagem.size)
transformacao = np.cumsum(probabilidades) * 255

imagem_equalizada = np.interp(imagem.flatten(), np.arange(256), transformacao) \
    .reshape(imagem.shape) \
    .astype(np.uint8)
histograma_equalizado, _ = np.histogram(imagem_equalizada.flatten(), 256, [0, 256])

plt.bar(np.arange(256), histograma)
plt.title('Histograma')
plt.xlabel('Nível de cinza')
plt.legend(['Original'])
plt.show()

plt.bar(np.arange(256), histograma_equalizado)
plt.title('Histograma Equalizado')
plt.xlabel('Nível de cinza')
plt.legend(['Original'])
plt.show()

plt.step(np.arange(256), transformacao)
plt.title('Função de Transformação')
plt.xlabel('Nível de cinza')
plt.legend(['Original'])
plt.show()

cv.namedWindow(f'{caminho}', cv.WINDOW_NORMAL)
cv.resizeWindow(f'{caminho}', 600, 600)
cv.imshow(f'{caminho}', imagem_equalizada)
cv.waitKey(0)
cv.destroyAllWindows()