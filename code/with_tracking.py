import cv2
import numpy as np
from utils.frame_processing import *

def find_contours(frame, filtered, centers, max_dist, next_id, frames_confirm, tracking):

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
                else:
                    # Criar um novo contorno
                    current_centers[next_id] = {'center': center, 'frames_visible': 1}
                    next_id += 1
                    
    centers = current_centers

    # Mostrar os centros detectados
    for contour_id in centers.keys():
        if centers[contour_id]['frames_visible'] >= frames_confirm and contour_id not in tracking:
            if not any(euclidean_distance(centers[contour_id]['center'], center[0]) <= max_dist for center in tracking.values()):
                tracking[contour_id] = (centers[contour_id]['center'], False)
         
    return frame, centers, next_id, tracking

def track(prev_frame, frame, original_frame, tracking: dict):

    ids_to_remove = []  # Lista para armazenar IDs que perderam o tracking

    for contour_id, t in tracking.items():
        if t[1] == True:
            prev_point = np.array([[t[0]]], dtype=np.float32)
            # Calcular o novo ponto com Optical Flow
            new_point,status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_point, None)
        
            if status[0] == 1:  # Se o tracking foi bem-sucedido
                new_x, new_y = new_point[0][0]
                tracking[contour_id] = ((new_x, new_y), True)
                cv2.circle(original_frame, (int(new_x), int(new_y)), 5, (0, 0, 255), -1)
                cv2.putText(original_frame, f"Tracking ID {contour_id}", (int(new_x), int(new_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:  # Caso o tracking seja perdido
                ids_to_remove.append(contour_id)
        else:
            tracking[contour_id] = (t[0], True)

    # Remover os IDs que perderam o tracking
    for contour_id in ids_to_remove:
        del tracking[contour_id]
        
    return original_frame, tracking

if __name__ == '__main__':
    # Carregar o vídeo
    cap = cv2.VideoCapture('peixe.MP4')

    # Inicialização de variáveis
    ret, prev_frame = cap.read()
    
    prev_frame_bw, prev_frame = prep_frame(prev_frame, 1)

    max_dist = 50  # Distância máxima para considerar o mesmo contorno
    centers = {}
    next_id = 0
    frames_confirm = 3
    tracking = {} # id, point
    processed_frames = []

    print("Processando vídeo")

    #Processar todos os frames e armazenar
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_bw, original_frame = prep_frame(frame, 1)
        frame = original_frame.copy()

        frame_transformed, _ = motion_compensation_v2(prev_frame_bw, frame_bw, prev_frame, frame)

        filtered = filtering(frame_transformed, frame_bw)
        
        original_frame, centers, next_id, tracking = find_contours(original_frame, filtered, centers, max_dist, next_id, frames_confirm, tracking)

        original_frame, tracking = track(prev_frame, frame, original_frame, tracking)
        
        processed_frames.append(original_frame)

        prev_frame = frame
        prev_frame_bw = frame_bw

        
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

