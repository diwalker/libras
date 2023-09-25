import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicializa o modelo do MediaPipe para detectar mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Carrega o modelo Keras treinado para reconhecimento de letras
model = tf.keras.models.load_model('keras_model.h5')

# Defina a lista de classes para mapear as previsões do modelo
classes = ['A', 'B', 'C', 'D', 'E']

# Loop principal para processar os quadros de vídeo
while True:
    # Captura o quadro da webcam
    success, img = cap.read()
    if not success:
        break

    # Converte a imagem para RGB (MediaPipe requer imagens RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar mãos
    results = hands.process(img_rgb)

    # Verifica se mãos foram detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Coleta as coordenadas dos pontos da mão
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            # Calcula os limites da caixa ao redor da mão
            x_min = int(min(landmarks, key=lambda x: x[0])[0] * img.shape[1])
            y_min = int(min(landmarks, key=lambda x: x[1])[1] * img.shape[0])
            x_max = int(max(landmarks, key=lambda x: x[0])[0] * img.shape[1])
            y_max = int(max(landmarks, key=lambda x: x[1])[1] * img.shape[0])

            # Recorta a região da mão e redimensiona para o tamanho esperado pelo modelo
            hand_crop = img[y_min:y_max, x_min:x_max]
            hand_crop = cv2.resize(hand_crop, (224, 224))

            # Normaliza a imagem e prepara os dados para previsão
            hand_array = (hand_crop.astype(np.float32) / 127.0) - 1
            data = np.expand_dims(hand_array, axis=0)

            # Realiza a previsão com o modelo Keras
            prediction = model.predict(data)
            indexVal = np.argmax(prediction)
            letter = classes[indexVal]

            # Desenha a caixa ao redor da mão e exibe a letra reconhecida
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe o quadro de vídeo
    cv2.imshow('Libras', img)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos da câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
