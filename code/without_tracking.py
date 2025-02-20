import cv2
import numpy as np
from utils.frame_processing import *
import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '8192' 

def find_contours(frame, filtered, centers, max_dist, next_id, frames_confirm):
    # Encontrar contornos na máscara sem as áreas conectadas às bordas
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Critérios para filtragem
    min_area = 250  # Tamanho mínimo do contorno

    current_centers = {}

    # Filtrar contornos
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:

            # Calcular o centroide
            M = cv2.moments(contour)
            if M['m00'] > 0:  # Evitar divisão por zero
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                center = (cX, cY)

                # Verificar se o contorno já estava sendo rastreado
                matched_id = None
                for contour_id, data in centers.items():
                    prev_center = data['center']
                    if euclidean_distance(center, prev_center) <= max_dist:
                        matched_id = contour_id
                        break

                if matched_id is not None:
                    # Atualizar o contorno existente
                    centers[matched_id]['center'] = center
                    centers[matched_id]['frames_visible'] += 1
                    current_centers[matched_id] = centers[matched_id]

                     # Pintar o contorno correspondente na imagem
                    if centers[matched_id]['frames_visible'] >= frames_confirm:
                        cv2.drawContours(frame, [contour], -1, (0, 0,255), thickness=cv2.FILLED)
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)  # Green centroid
                else:
                    # Criar um novo contorno
                    current_centers[next_id] = {'center': center, 'frames_visible': 1}
                    next_id += 1

    centers = current_centers

    return frame, centers, next_id

if __name__ == '__main__':
    # Carregar o vídeo
    cap = cv2.VideoCapture('peixe.MP4')

    if (cap.isOpened()== False): 
        print("Erro ao abrir o vídeo.")
        exit()

    # Inicialização de variáveis
    ret, prev_frame = cap.read()
    prev_frame = prep_frame(prev_frame)

    max_dist = 100  # Distância máxima para considerar o mesmo contorno
    centers = {}
    next_id = 0
    frames_confirm = 3
    processed_frames = []

    print("Processando vídeo")

    #Processar todos os frames e armazenar
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, original_frame = prep_frame(frame, 1)
        frame_hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
        frame_transformed, _ = motion_compensation(prev_frame, frame)
        filtered = filtering(frame_transformed, frame)
        
        original_frame, centers, next_id = find_contours(original_frame, filtered, centers, max_dist, next_id, frames_confirm)

        prev_frame = frame
        processed_frames.append(original_frame)

    cap.release()
    input(f"\nProcessamento concluído: {len(processed_frames)} frames processados.\nPrecione enter tecla para reproduzir o vídeo\n")
    
    #Reproduzir os frames processados

    frame_interval = 1 / 10

    for i,frame in enumerate(processed_frames):
        cv2.imshow('Processed', frame)
        
        # Aguardar o tempo necessário para manter 30 FPS
        if cv2.waitKey(int(frame_interval * 1000)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()