import cv2
import numpy as np
import json
import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '8192' 

def morphology(frame):

    # 1º juntar as areas muito proximas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    # 2ª remover as area pequenas, provavelmente são noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

    # 3ª juntar as areas que estão relativamente perto melhor visualização
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    return frame 

def algea_contours(frame, filtered):  
    # Encontrar contornos na máscara 
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tamanho mínimo do contorno
    min_area = 500  

    # Filtrar contornos
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
    
    return frame

def load_limits_from_json(json_file):
    #Carregar limites de H, S, V a partir de um arquivo JSON
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            if "limits" in data:
                min_lim = []
                max_lim = []
                for limit in data["limits"]:
                    min_lim.append(np.array([limit["H"]["min"], limit["S"]["min"], limit["V"]["min"]]))
                    max_lim.append(np.array([limit["H"]["max"], limit["S"]["max"], limit["V"]["max"]]))
                return min_lim, max_lim
    return None, None

if __name__ == '__main__':
    # Carregar o vídeo
    cap = cv2.VideoCapture('peixe.mp4')

    if (cap.isOpened()== False): 
        print("Erro ao abrir o vídeo.")
        exit()
    
    # Nome do arquivo JSON com os limites
    json_file = 'limits.json'

    # Carregar limites do JSON, se existir
    min_lim, max_lim = load_limits_from_json(json_file)

    # Definir limites padrão caso o JSON não exista
    if not min_lim or not max_lim:
        min_lim = [
            np.array([30, 50, 65]),
            np.array([40, 50, 65]),       
            np.array([50, 50, 65]),
            np.array([65, 50, 65]),
        ]

        max_lim = [
            np.array([40, 255, 255]),
            np.array([50, 255, 255]),
            np.array([65, 255, 255]),
            np.array([75, 255, 255]),
        ]

    # Exibir os limites carregados ou padrão
    print("╭──────────────────────────╮")
    print("│----- Limites a usar -----│")
    print("├───┬──────────┬───────────┤")
    for i, (l_min, l_max) in enumerate(zip(min_lim, max_lim)):
        print(f"│ H │ mín: {l_min[0]:3} │ máx: {l_max[0]:3}  │")
        print(f"│ S │ mín: {l_min[1]:3} │ máx: {l_max[1]:3}  │")
        print(f"│ V │ mín: {l_min[2]:3} │ máx: {l_max[2]:3}  │")
        if i+1 < len(min_lim):
            print(f"├───┼──────────┼───────────┤")
        else:
            print(f"└───┴──────────┴───────────┘")
            
    # Inicialização de variáveis
    width = 1280
    height = 720
    processed_frames = []

    print("Processando vídeo")

    #Processar todos os frames e armazenar
    while True:
        ret, frame = cap.read()
        if not ret:             
            break
        frame = cv2.resize(frame, (width, height))

        # Separar os canais para equalizar
        b,g,r = cv2.split(frame)

        # Equalizar todos os canais para aumentar o contraste
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        # Combinar os canais novamente
        frame_eq = cv2.merge((b_eq, g_eq, r_eq))

        # Converter a frame para HSV para melhor detetar as cores pertendidas
        frame_hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)

        # Percorrer por todos os limites defenido e a criar a mascara com apenas os pixeis que constam nesses valor e juntar todas as mascaras
        combined_mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        for l_min, l_max in zip(min_lim, max_lim):
            mask = cv2.inRange(frame_hsv, l_min, l_max)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Filtragem morfológica 
        final_mask = morphology(combined_mask)

        # Aplicar contornos
        frame_eq = algea_contours(frame_eq, final_mask)

        # Guardar os frames para apresentar depois
        processed_frames.append(frame_eq)     
  
    cap.release()
    input(f"\nProcessamento concluído: {len(processed_frames)} frames processados.\nPrecione enter para reproduzir o vídeo\n")
    
    #Reproduzir os frames processados a 30 FPS
    frame_interval = 1 / 30  

    for i, frame in enumerate(processed_frames):
        cv2.imshow('Processed', frame)
        if i==1:
            cv2.waitKey(0)
        # Aguardar o tempo necessário para manter 30 FPS
        if cv2.waitKey(int(frame_interval * 1000)) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
