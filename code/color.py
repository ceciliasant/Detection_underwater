#!/usr/bin/env python3
import cv2
import json
from copy import deepcopy
import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '8192' 

# Função para capturar os clicks
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        param['click_position'] = (x, y)  # Salva a posição clicada

def onTrackbar(threshold, mm, C, limits):
    limits[C][mm] = threshold

def info():
    print(f"╭────────────────────────────────────────────────────────────────────────╮")
    print(f"│--------------------------  Comandos Possíveis -------------------------│")
    print(f"├─────┬──────────────────────────────────────────────────────────────────┤")
    print(f"│ n N │ Avançar para o proximo frame                                     │")
    print(f"├─────┼──────────────────────────────────────────────────────────────────┤")
    print(f"│ l L │ Registar os intervalos atuais de HSV para gravar                 │")
    print(f"├─────┼──────────────────────────────────────────────────────────────────┤")
    print(f"│ w W │ Gravar no ficheiro limits.json todos os intervalos HSV guardados │")
    print(f"├─────┼──────────────────────────────────────────────────────────────────┤")
    print(f"│ q Q │ Fechar as janelas e fechar o programa                            │")
    print(f"└─────┴──────────────────────────────────────────────────────────────────┘")
    return

# Recolher os valores para a segmentação de cor
def main():
    mask_window = "Mask"
    frame_window = "Original"
    limits_save = []

    # Criar janelas separadas
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mask_window, 800, 680)  
    cv2.namedWindow(frame_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(frame_window, 800, 550)  

    cap = cv2.VideoCapture('peixe.mp4')

    if (cap.isOpened()== False): 
        print("Erro ao abrir o vídeo.")
        exit()

    limits = {"H": {"max": 179, "min": 0}, "S": {"max": 255, "min": 0}, "V": {"max": 255, "min": 0}}
    width, height = 640, 480
    click_data = {'click_position': None}  # Variável compartilhada para salvar a posição do clique

    # Criar sliders na janela da máscara
    cv2.createTrackbar("H-min", mask_window, limits["H"]["min"], 179, lambda threshold: onTrackbar(threshold, mm="min", C="H", limits=limits))
    cv2.createTrackbar("H-max", mask_window, limits["H"]["max"], 179, lambda threshold: onTrackbar(threshold, mm="max", C="H", limits=limits))
    cv2.createTrackbar("S-min", mask_window, limits["S"]["min"], 255, lambda threshold: onTrackbar(threshold, mm="min", C="S", limits=limits))
    cv2.createTrackbar("S-max", mask_window, limits["S"]["max"], 255, lambda threshold: onTrackbar(threshold, mm="max", C="S", limits=limits))
    cv2.createTrackbar("V-min", mask_window, limits["V"]["min"], 255, lambda threshold: onTrackbar(threshold, mm="min", C="V", limits=limits))
    cv2.createTrackbar("V-max", mask_window, limits["V"]["max"], 255, lambda threshold: onTrackbar(threshold, mm="max", C="V", limits=limits))
    
    # Callback para detectar cliques na máscara
    cv2.setMouseCallback(mask_window, click, click_data)
    
    info()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo.")
            break
        
        # Equalização e redimensionamento
        b, g, r = cv2.split(frame)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        frame = cv2.merge((b_eq, g_eq, r_eq))
        frame = cv2.resize(frame, (width, height))

        while True:  # Atualização do mesmo frame
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame,
                               (limits["H"]["min"], limits["S"]["min"], limits["V"]["min"]),
                               (limits["H"]["max"], limits["S"]["max"], limits["V"]["max"]))

            # Adicionar marcador na imagem original, caso clique seja detectado
            frame_with_marker = frame.copy()
            if click_data['click_position']:
                click_x, click_y = click_data['click_position']
                cv2.circle(frame_with_marker, (click_x, click_y), 5, (0, 0, 255), -1)
                click_data['click_position'] = None 

            # Exibir as janelas separadas
            cv2.imshow(mask_window, mask)
            cv2.imshow(frame_window, frame_with_marker)

            k = cv2.waitKey(1) 
            if k == ord("q") or k == ord("Q"):  # Sair
                cap.release()
                cv2.destroyAllWindows()
                return
            
            elif k == ord("l") or k == ord("L"):  # Guardar os numa lista
                limits_save.append(deepcopy(limits))

            elif k == ord("w") or k == ord("W"):  # Guardar os limites na lista
                file_name = "limits.json"
                with open(file_name, 'w') as json_file:
                    for n,l in enumerate(limits_save):
                        if n == 0:
                            print("           VALUES        ")
                            print(f"╭───┬──────────┬──────────╮")
                        print(f"│ H │ min: {l['H']['min']:3} │ max: {l['H']['max']:3} │")
                        print(f"│ S │ min: {l['S']['min']:3} │ max: {l['S']['max']:3} │")
                        print(f"│ V │ min: {l['V']['min']:3} │ max: {l['V']['max']:3} │")
                        if n+1 < len(limits_save):
                            print(f"├───┼──────────┼──────────┤")
                        else:
                            print(f"└───┴──────────┴──────────┘")
                    json.dump({"limits": limits_save}, json_file, indent=4)

            elif k == ord("n") or k == ord("N"):  # Passar para o próximo frame
                break
            



if __name__ == '__main__':
    main()
