# Importando biblioteca, IA e framework
import cv2 
from ultralytics import YOLO

import time
import numpy as np

# Carregando arquivos de peso
modelo = YOLO("yolov8n-seg.pt")

camera = cv2.VideoCapture(0)

# Lista para a ia identificar e soltar alerta
lista_interesses = ["person", "car", "bus", "truck"]

# Para calcular o fps
tempo_anterior = 0

cor_verde = (0, 255, 0)

while True:
    # Lendo entrada da camera
    sucesso, frame = camera.read()

    # A IA redimensiona a imagem, converte para RGB, faz a inferência e devolve um pacote com coordenadas e classes.
    # E diminui o tamanho da imagem para ter um melhor processamento.
    resultados = modelo(frame, imgsz=320)

    # Desenhando caixa magica(x,y,w,h) por meio do framework
    frame_anotado = resultados[0].plot()

    # 1. Primeiro verificamos se a IA detectou alguma máscara neste milissegundo
    if resultados[0].masks is not None:
        
        # 2. Extraímos a lista de pontos matemáticos (x, y) do PRIMEIRO objeto que ela viu
        contorno_objeto = resultados[0].masks.xy[0]
        
        # 1. A sintaxe [:, 0] é o bisturi do NumPy: ela extrai apenas a Coluna X de todos os  pontos.
        eixo_x = contorno_objeto[:, 0]
        # 2. Achamos os extremos instantaneamente
        x_min = np.min(eixo_x)
        x_max = np.max(eixo_x)

        # 3. Calculamos o centro exato da máscara irregular
        centro_mascara = int((x_min + x_max) / 2)
        
        print(f"O centro real da superfície é: {centro_mascara}")

    # Função para captura de tempo exato do ms para calcular o fps
    tempo_atual = time.time()

    # Formula para calcular fps
    fps = 1 / (tempo_atual - tempo_anterior)

    # Atualizando o valor do tempo
    tempo_anterior = tempo_atual

    # Analisando um objeto por vez
    for caixa in resultados[0].boxes:
        # Descompactando para conseguir as informaçoes com as coordenadas das caixas
        x1,y1,x2,y2 = caixa.xyxy[0]

        # Descobrindo nome dos objetos
        nome_objeto = modelo.names[int(caixa.cls[0])]

        # Descobrindo o centro da caixa
        centro_horizontal_caixa = int((x1 + x2) / 2)

        # Confiança na definição do objeto
        confianca = int(caixa.conf[0] * 100)

        # Criando avisos
        if confianca >= 75 and nome_objeto in lista_interesses:
            if centro_horizontal_caixa <= 200:
                print(f"Atenção: {nome_objeto} a esquerda!")
            elif centro_horizontal_caixa > 200 and centro_horizontal_caixa <= 400:
                print(f"PERIGO: {nome_objeto} no meio do caminho")
            else:
                print(f"Atenção: {nome_objeto} a direita")
        else:
            # Comando para ignorar quando não tiver certeza
            pass

    # Exibindo caixa de texto com o fps que pegamos
    cv2.putText(frame_anotado, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_verde, 2)
    cv2.imshow("Camera com uso do YOLO", frame_anotado)

    tecla = cv2.waitKey(1)
    if tecla == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
