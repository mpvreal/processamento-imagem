'''
Aluno: Matheus Pires Vila Real    Matrícula: 202000560352
######### Exercício 5 #########
'''

import cv2
import numpy as np

from sys import argv

# Função para exibir imagens
def mostrar_imagem(titulo, imagem):
  cv2.imshow(titulo, imagem)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Função para aplicar a abertura (erosão seguida de dilatação)
def abertura(imagem, elemestrut):
  erosao = cv2.erode(imagem, elemestrut, iterations=1)
  abertura_resultado = cv2.dilate(erosao, elemestrut, iterations=1)
  return abertura_resultado

# Função para aplicar o fechamento (dilatação seguida de erosão)
def fechamento(imagem, elemestrut):
  dilatacao = cv2.dilate(imagem, elemestrut, iterations=1)
  fechamento_resultado = cv2.erode(dilatacao, elemestrut, iterations=1)
  return fechamento_resultado

# Função para extrair as fronteiras
def extrair_fronteiras(imagem, elemestrut):
  fronteiras = cv2.absdiff(imagem, cv2.erode(imagem, elemestrut, iterations=1))
  return fronteiras

# Função para preencher buracos
def preencher_buracos(imagem, x, elemestrut):
  # Complemento da imagem original
  imagem_complemento = cv2.bitwise_not(imagem)
  # Buffer para verificar se houve mudança
  x_k_menos_1 = np.zeros_like(x)

  while not np.array_equal(x_k_menos_1, x): # Só para quando não houver mais mudanças
    x_k_menos_1 = x.copy()
    # Dilatação de X_{k-1}
    imagem_dilatada = cv2.dilate(x, elemestrut, iterations=1)
    # Interseção da imagem dilatada com a imagem original
    x = cv2.bitwise_and(imagem_dilatada, imagem_complemento)
  
  return x

# Função para extrair componentes conexos
def extrair_componentes_conexos(imagem, x, elemestrut):
  # Buffer para verificar se houve mudança
  x_k_menos_1 = np.zeros_like(x)

  while not np.array_equal(x_k_menos_1, x): # Só para quando não houver mais mudanças
    x_k_menos_1 = x.copy()
    # Dilatação de X_{k-1}
    imagem_dilatada = cv2.dilate(x, elemestrut, iterations=1)
    # Interseção da imagem dilatada com a imagem original
    x = cv2.bitwise_and(imagem_dilatada, imagem)
  
  return x

try:
  imagem = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
except IndexError:
  print(f'Modo de utilização:\n    $ python {argv[0]} <caminho-para-imagem>')

_, imagem = cv2.threshold(imagem, 128, 255, cv2.THRESH_BINARY)

# Elemento estruturante
elemestrut = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Aplicar as operações e exibir os resultados
mostrar_imagem('Imagem Original', imagem)

resultado_abertura = abertura(imagem, elemestrut)
mostrar_imagem('Abertura', resultado_abertura)

resultado_fechamento = fechamento(imagem, elemestrut)
mostrar_imagem('Fechamento', resultado_fechamento)

resultado_fronteiras = extrair_fronteiras(imagem, elemestrut)
mostrar_imagem('Fronteiras', resultado_fronteiras)

x0 = np.zeros_like(imagem)
x0[0, 0] = 255

resultado_preenchimento_buracos = preencher_buracos(imagem, x0, elemestrut)
mostrar_imagem('Preenchimento de Buracos', resultado_preenchimento_buracos)

resultado_componentes_conexos = extrair_componentes_conexos(imagem, x0, elemestrut)
mostrar_imagem('Componentes Conexos', resultado_componentes_conexos)
