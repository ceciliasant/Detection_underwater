import cv2
import numpy as np

def histo(frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histSize = 256	 # from 0 to 255
    # Intensity Range
    histRange = [0, 256]

    # Compute the histogram
    hist_item = cv2.calcHist([gray], [0], None, [histSize], histRange)

    # Create an image to display the histogram
    histImageWidth = 512
    histImageHeight = 512
    color = (125)
    histImage = np.zeros((histImageWidth,histImageHeight,1), np.uint8)

    # Width of each histogram bar
    binWidth = int (np.ceil(histImageWidth*1.0 / histSize))

    # Normalize values to [0, histImageHeight]
    cv2.normalize(hist_item, hist_item, 0, histImageHeight, cv2.NORM_MINMAX)

    # Draw the bars of the nomrmalized histogram
    for i in range (histSize):
        cv2.rectangle(histImage,  ( i * binWidth, 0 ), ( ( i + 1 ) * binWidth, int(hist_item[i]) ), color, -1)

    # ATTENTION : Y coordinate upside down
    histImage = np.flipud(histImage)

    cv2.imshow(name, histImage)

def motion_compensation(prev_frame, curr_frame):

    # Detecção de pontos de característica usando Shi-Tomasi
    prev_points = cv2.goodFeaturesToTrack(prev_frame, maxCorners=1000, qualityLevel=0.01, minDistance=30)
    # print(prev_points)
    # Cálculo do fluxo óptico usando Lucas-Kanade
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None)

    # Seleção de pontos válidos
    prev_points = prev_points[status == 1]
    curr_points = curr_points[status == 1]

    # Estimação da transformação
    transform_matrix, mask = cv2.estimateAffinePartial2D(prev_points, curr_points)
    frame1_transformed = cv2.warpAffine(prev_frame, transform_matrix, (curr_frame.shape[1], curr_frame.shape[0]))

    return frame1_transformed, mask

def motion_compensation_v2(prev_frame_bw, curr_frame_bw, prev_frame, curr_frame):

    # Detecção de pontos de característica usando Shi-Tomasi
    prev_points = cv2.goodFeaturesToTrack(prev_frame_bw, maxCorners=1000, qualityLevel=0.01, minDistance=30)

    # Cálculo do fluxo óptico usando Lucas-Kanade
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None)

    # Seleção de pontos válidos
    prev_points = prev_points[status == 1]
    curr_points = curr_points[status == 1]

    # Estimação da transformação
    transform_matrix, mask = cv2.estimateAffinePartial2D(prev_points, curr_points)
    frame1_transformed = cv2.warpAffine(prev_frame_bw, transform_matrix, (curr_frame.shape[1], curr_frame.shape[0]))

    return frame1_transformed, mask

# Função para calcular a distância euclidiana
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def filtering(frame_transformed, frame):
     # Diferença entre o frame compensado e o frame atual
    diff_frame = cv2.absdiff(frame_transformed, frame)
   
    # Conversão para escala de cinza e limiarização
    _, thresh_diff = cv2.threshold(diff_frame, 70, 255, cv2.THRESH_BINARY)

    # Remoção de ruídos com operações morfológicas e juntar "brancos" proximos
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))

    clean_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, kernel1)
    clean_diff = cv2.morphologyEx(clean_diff, cv2.MORPH_CLOSE, kernel2)

    h, w = clean_diff.shape

    # Definir a espessura das bordas a serem preenchidas
    border_thickness = 15  

    # Preencher as bordas com retângulos
    mask_no_borders = cv2.rectangle(clean_diff, (0, 0), (w - 1, border_thickness - 1), 0, -1)  # Borda superior
    mask_no_borders = cv2.rectangle(mask_no_borders, (0, h - border_thickness), (w - 1, h - 1), 0, -1)  # Borda inferior
    mask_no_borders = cv2.rectangle(mask_no_borders, (0, 0), (border_thickness - 1, h - 1), 0, -1)  # Borda esquerda
    mask_no_borders = cv2.rectangle(mask_no_borders, (w - border_thickness, 0), (w - 1, h - 1), 0, -1)  # Borda direita
    return mask_no_borders

def prep_frame(frame, save_copy = 0):
    width = 1280
    height = 720
    frame = cv2.resize(frame, (width, height))

    b,g,r = cv2.split(frame)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    frame = cv2.merge((b_eq, g_eq, r_eq))

    if save_copy:
        copy = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    frame = cv2.medianBlur(frame, 11)
    if save_copy:
        return frame, copy
    return frame
