#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Face Counter & Emotion Analyzer - Uma aplicação para contagem e análise emocional de faces
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import datetime
import time
import json
from collections import defaultdict, Counter
import traceback

# Configurar logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(
    log_dir, f"face_counter_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("FaceCounter")

# Registrar início da aplicação
logger.info("Iniciando aplicativo Face Counter & Emotion Analyzer")

# Tente importar deepface - pode falhar se todas as dependências não estiverem instaladas
try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace carregado com sucesso")
except ImportError as e:
    logger.warning(f"DeepFace não está completamente disponível: {e}")
    DEEPFACE_AVAILABLE = False


class FaceCounter:
    """Classe para detectar, reconhecer e contar faces em imagens"""

    def __init__(self):
        # Configurações iniciais
        self.face_features_db = {}
        self.tracked_faces = {}
        self.next_temp_id = 1
        self.known_faces = {}
        self.start_time = time.time()

        # Estatísticas
        self.unique_faces_detected = 0
        self.total_faces_detected = 0
        self.emotion_stats = {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "neutral": 0,
            "unknown": 0,
        }

        # Estatísticas de idade
        self.age_stats = {}

        # Janela deslizante para emoções recentes
        self.recent_emotions = []
        self.recent_window_size = 30
        self.last_emotion_update = time.time()

        # Configurações para reconhecimento
        self.recognition_threshold = 0.3
        self.min_face_size = (30, 30)
        self.last_cleanup = time.time()
        self.face_tracking_threshold = 3.0

        # Pasta para salvar faces conhecidas
        self.known_faces_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "images", "known_faces"
        )
        os.makedirs(self.known_faces_dir, exist_ok=True)

        # Pasta para imagens
        self.images_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "images"
        )
        os.makedirs(self.images_dir, exist_ok=True)

        # Pasta para output
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(
            "Contador de faces inicializado com threshold de reconhecimento: %.2f",
            self.recognition_threshold,
        )

    def load_known_faces(self):
        """Carrega as faces conhecidas do arquivo JSON"""
        faces_json_path = os.path.join(self.known_faces_dir, "known_faces.json")
        if os.path.exists(faces_json_path):
            try:
                with open(faces_json_path, "r") as f:
                    self.known_faces = json.load(f)
                logger.info(f"Carregadas {len(self.known_faces)} faces conhecidas")
            except Exception as e:
                logger.error(f"Erro ao carregar faces conhecidas: {e}")
                self.known_faces = {}
        else:
            logger.info(
                "Nenhuma face conhecida encontrada. Criando novo banco de dados."
            )
            self.known_faces = {}

    def save_known_faces(self):
        """Salva as faces conhecidas no arquivo JSON"""
        faces_json_path = os.path.join(self.known_faces_dir, "known_faces.json")
        try:
            with open(faces_json_path, "w") as f:
                json.dump(self.known_faces, f, indent=4)
            logger.info(f"Salvadas {len(self.known_faces)} faces conhecidas")
        except Exception as e:
            logger.error(f"Erro ao salvar faces conhecidas: {e}")

    def register_new_face(self, face_img, emotion=None, age=None):
        """Registra uma nova face na base de dados de faces conhecidas"""
        try:
            # Gerar um ID único para esta face
            self.unique_faces_detected += 1
            face_id = f"face_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.unique_faces_detected}"

            # Salvar imagem da face
            face_img_path = os.path.join(self.known_faces_dir, f"{face_id}.jpg")
            cv2.imwrite(face_img_path, face_img)

            # Nome padrão
            default_name = f"Pessoa {self.unique_faces_detected}"

            # Adicionar ao banco de dados
            self.known_faces[face_id] = {
                "name": default_name,
                "image_path": face_img_path,
                "first_seen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "detection_count": 1,
                "emotions": {emotion: 1} if emotion else {},
                "ages": [age] if age is not None else [],
            }

            # Adicionar também à estatística global de idade
            if age is not None:
                if face_id not in self.age_stats:
                    self.age_stats[face_id] = []
                self.age_stats[face_id].append(age)

            # Salvar base de dados atualizada
            self.save_known_faces()

            logger.info(f"Nova face registrada: ID={face_id}, Nome={default_name}")

            # Adicionar à estatística de emoções se fornecida
            if emotion:
                self.emotion_stats[emotion] += 1

            return face_id, default_name

        except Exception as e:
            logger.error(f"Erro ao registrar nova face: {e}")
            return None, None

    def compare_faces(self, face_encoding1, face_encoding2):
        """
        Comparar duas codificações faciais para determinar se são a mesma pessoa
        Retorna True se as faces corresponderem à mesma pessoa, False caso contrário
        """
        if face_encoding1 is None or face_encoding2 is None:
            return False

        # Calcular distância euclidiana entre as características
        distance = np.linalg.norm(np.array(face_encoding1) - np.array(face_encoding2))

        # Quanto menor a distância, mais similar são as faces
        return distance < self.recognition_threshold

    def is_same_as_known_face(self, face_encoding):
        """
        Verificar se a codificação facial corresponde a alguma face já conhecida
        Retorna o ID da face conhecida se encontrada, None caso contrário
        """
        for face_id, known_encoding in self.face_features_db.items():
            if self.compare_faces(face_encoding, known_encoding):
                return face_id
        return None

    def recognize_face(self, face_img, emotion=None, age=None):
        """Reconhece uma face e retorna seu ID e nome, ou registra como nova se não for reconhecida"""
        if not DEEPFACE_AVAILABLE:
            return None, None

        try:
            # Verificar se esta face corresponde a alguma face conhecida
            for face_id, face_data in self.known_faces.items():
                image_path = face_data["image_path"]

                # Verificar se o arquivo existe
                if not os.path.exists(image_path):
                    continue

                # Comparar faces
                try:
                    result = DeepFace.verify(
                        img1_path=face_img,
                        img2_path=image_path,
                        enforce_detection=False,
                        model_name="VGG-Face",
                        distance_metric="cosine",
                    )

                    # Se a verificação for bem-sucedida e as faces corresponderem
                    if result["verified"]:
                        # Atualizar estatísticas da face
                        self.known_faces[face_id][
                            "last_seen"
                        ] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.known_faces[face_id]["detection_count"] += 1

                        # Atualizar contagem de emoções
                        if emotion:
                            if "emotions" not in self.known_faces[face_id]:
                                self.known_faces[face_id]["emotions"] = {}

                            if emotion in self.known_faces[face_id]["emotions"]:
                                self.known_faces[face_id]["emotions"][emotion] += 1
                            else:
                                self.known_faces[face_id]["emotions"][emotion] = 1

                        # Armazenar idade detectada
                        if age is not None:
                            if "ages" not in self.known_faces[face_id]:
                                self.known_faces[face_id]["ages"] = []

                            # Adicionar idade à lista de idades detectadas para esta pessoa
                            self.known_faces[face_id]["ages"].append(age)

                            # Armazenar também no dicionário global de estatísticas de idade
                            if face_id not in self.age_stats:
                                self.age_stats[face_id] = []
                            self.age_stats[face_id].append(age)

                        # Salvar alterações
                        self.save_known_faces()

                        return face_id, self.known_faces[face_id]["name"]

                except Exception as e:
                    logger.debug(f"Erro ao comparar faces: {e}")
                    continue

            # Se chegou aqui, nenhuma face conhecida foi correspondida
            return None, None

        except Exception as e:
            logger.error(f"Erro ao reconhecer face: {e}")
            return None, None

    def update_recent_emotion(self, emotion):
        """Atualiza a lista de emoções recentes e retorna a emoção predominante atual"""
        if not emotion:
            return None

        # Adicionar a emoção atual à lista de recentes
        self.recent_emotions.append(emotion)

        # Limitar o tamanho da janela deslizante
        if len(self.recent_emotions) > self.recent_window_size:
            self.recent_emotions.pop(0)  # Remover a emoção mais antiga

        # Atualizar o timestamp da última atualização
        self.last_emotion_update = time.time()

        # Calcular a emoção predominante na janela recente
        if self.recent_emotions:
            # Usar Counter para encontrar a emoção mais frequente na janela
            emotion_counts = Counter(self.recent_emotions)
            predominant_emotion = emotion_counts.most_common(1)[0][0]
            return predominant_emotion

        return None

    def process_frame(self, frame):
        """
        Processar um frame para detectar, reconhecer e analisar faces
        """
        if frame is None:
            return frame

        # Criar uma cópia para desenhar
        display_frame = frame.copy()

        # Resetar faces rastreadas para o frame atual
        self.tracked_faces = {}

        try:
            # Detectar faces usando DeepFace
            faces = DeepFace.extract_faces(
                img_path=frame,
                target_size=(224, 224),
                detector_backend="opencv",
                enforce_detection=False,
            )

            for i, face_data in enumerate(faces):
                # Obter região da face (expandimos um pouco para melhor análise)
                face_region = face_data["facial_area"]
                x, y, w, h = (
                    face_region["x"],
                    face_region["y"],
                    face_region["w"],
                    face_region["h"],
                )

                # Ignorar faces muito pequenas
                if w < self.min_face_size[0] or h < self.min_face_size[1]:
                    continue

                # Expandir a região da face para melhor análise (com limites seguros)
                safe_x = max(0, x - int(w * 0.1))
                safe_y = max(0, y - int(h * 0.1))
                safe_w = min(frame.shape[1] - safe_x, int(w * 1.2))
                safe_h = min(frame.shape[0] - safe_y, int(h * 1.2))

                face_img = frame[safe_y : safe_y + safe_h, safe_x : safe_x + safe_w]

                # Análise facial completa para obter emoções e vetores de características
                try:
                    analysis = DeepFace.analyze(
                        img_path=face_img,
                        actions=["emotion", "embedding"],
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True,
                    )

                    if isinstance(analysis, list) and len(analysis) > 0:
                        analysis = analysis[0]

                    # Obter vetor de características faciais
                    face_encoding = analysis.get("embedding", None)
                    dominant_emotion = analysis.get("dominant_emotion", "unknown")

                    # Procurar por face correspondente no banco de dados
                    face_id = self.is_same_as_known_face(face_encoding)

                    if face_id is None:
                        # Nova face encontrada
                        face_id = f"person_{self.next_temp_id}"
                        self.next_temp_id += 1
                        self.face_features_db[face_id] = face_encoding
                        self.unique_faces_detected += 1

                        # Salvar nova face conhecida
                        if self.known_faces_dir:
                            face_file = os.path.join(
                                self.known_faces_dir, f"{face_id}.jpg"
                            )
                            cv2.imwrite(face_file, face_img)

                    # Registrar esta face como vista no frame atual
                    self.tracked_faces[face_id] = {
                        "position": (x, y, w, h),
                        "emotion": dominant_emotion,
                        "last_seen": time.time(),
                    }

                    # Incrementar estatísticas
                    self.total_faces_detected += 1
                    self.emotion_stats[dominant_emotion] += 1

                    # Atualizar emoção predominante recente
                    self.update_recent_emotion(dominant_emotion)

                    # Desenhar retângulo e informações
                    color = (0, 255, 0)  # Verde para face reconhecida
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                    # Mostrar ID e emoção
                    face_label = f"{face_id}: {dominant_emotion}"
                    cv2.putText(
                        display_frame,
                        face_label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                except Exception as e:
                    logging.warning(f"Erro ao analisar face {i}: {str(e)}")
                    continue

            # Limpeza de rastreadores antigos (a cada 5 segundos)
            current_time = time.time()
            if current_time - self.last_cleanup > 5:
                self.cleanup_trackers()
                self.last_cleanup = current_time

        except Exception as e:
            logging.error(f"Erro ao processar frame: {str(e)}")
            traceback.print_exc()

        return display_frame

    def cleanup_trackers(self):
        """Limpar rastreadores inativos"""
        inactive_ids = [
            face_id
            for face_id, face_data in self.tracked_faces.items()
            if time.time() - face_data["last_seen"] > self.face_tracking_threshold
        ]
        for face_id in inactive_ids:
            del self.tracked_faces[face_id]

    def detect_faces_webcam(self):
        """Detecta faces em tempo real usando a webcam com contagem de pessoas e análise emocional"""
        logger.info("Iniciando detecção facial pela webcam com análise emocional")

        try:
            # Carregar o classificador Haar Cascade para detecção facial
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            logger.debug("Classificador facial carregado com sucesso para webcam")

            # Iniciar captura de vídeo
            cap = cv2.VideoCapture(1)

            # Configurar resolução e FPS para melhor performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer para minimizar o delay

            if not cap.isOpened():
                logger.error("Erro: Não foi possível acessar a webcam")
                return

            logger.info(
                f"Webcam iniciada com sucesso. Resolução: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )
            print(
                "Iniciando detecção facial com contagem de pessoas e análise emocional..."
            )
            print(
                "Pressione 'q' para sair, 's' para salvar um frame, 'e' para ver estatísticas de emoções"
            )

            frame_count = 0
            can_analyze = DEEPFACE_AVAILABLE
            logger.info(f"Análise emocional: {'ATIVA' if can_analyze else 'INATIVA'}")

            # Estatísticas de desempenho
            total_frames = 0
            total_faces_detected = 0
            total_analysis_attempts = 0
            successful_analyses = 0
            start_time = time.time()
            last_fps_update = start_time
            fps = 0

            # Controle de frequência de análise para reduzir carga
            skip_frames = 1
            frame_idx = 0
            last_analysis_time = time.time()
            min_analysis_interval = 0.3

            # Configurar parâmetros para detecção facial
            min_face_size = (60, 60)
            scale_factor = 1.2
            min_neighbors = 6

            # Rastreamento de faces na sessão atual
            current_session_faces = {}

            # Inicializar contadores de emoções na sessão atual
            session_emotions = defaultdict(int)

            while True:
                # Capturar frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    logger.error("Erro: Falha ao capturar frame da webcam")
                    break

                frame_start = time.time()
                total_frames += 1
                frame_idx += 1

                # Atualizar FPS a cada segundo
                if time.time() - last_fps_update >= 1.0:
                    fps = total_frames / (time.time() - start_time)
                    last_fps_update = time.time()

                # Converter para escala de cinza
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                # Detectar faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_face_size,
                )

                total_faces_detected += len(faces)

                # Exibir FPS e contagens
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

                # Mostrar contagem de pessoas únicas
                cv2.putText(
                    frame,
                    f"Pessoas únicas identificadas: {len(self.known_faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Mostrar emoção predominante (usando a janela deslizante)
                if self.recent_emotions:
                    # Usar Counter para encontrar a emoção mais frequente na janela recente
                    emotion_counts = Counter(self.recent_emotions)
                    current_top_emotion = emotion_counts.most_common(1)[0][0]

                    cv2.putText(
                        frame,
                        f"Sentimento predominante: {current_top_emotion}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                elif (
                    self.emotion_stats
                ):  # Fallback para o método antigo se não houver emoções recentes
                    top_emotion = max(self.emotion_stats.items(), key=lambda x: x[1])[0]
                    cv2.putText(
                        frame,
                        f"Sentimento predominante: {top_emotion}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Para cada rosto detectado
                for face_index, (x, y, w, h) in enumerate(faces):
                    # Calcular área da face para filtrar
                    face_area = w * h
                    face_width_ratio = w / frame.shape[1]

                    # Filtrar faces muito pequenas ou muito grandes
                    if face_area < 3600 or face_width_ratio > 0.7:
                        continue

                    # Extrair a imagem da face
                    face_img = frame[y : y + h, x : x + w]

                    # Chave única para esta face baseada na posição
                    face_position_key = f"{x}_{y}_{w}_{h}"

                    # Analisar face se possível
                    if can_analyze and (
                        face_position_key not in current_session_faces
                        or time.time()
                        - current_session_faces[face_position_key].get("last_check", 0)
                        > 2.0
                    ):

                        # Tentar fazer análise emocional
                        emotion = None
                        age = None

                        if DEEPFACE_AVAILABLE:
                            try:
                                # Salvar temporariamente para análise
                                temp_path = os.path.join(
                                    self.images_dir, f"temp_face_{face_index+1}.jpg"
                                )
                                cv2.imwrite(temp_path, face_img)

                                # Analisar com DeepFace
                                result = DeepFace.analyze(
                                    temp_path,
                                    actions=["emotion", "age"],
                                    enforce_detection=False,
                                )

                                # Remover arquivo temporário
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

                                # Extrair informações
                                if isinstance(result, list):
                                    result = result[0]

                                if result:
                                    if "emotion" in result:
                                        emotion = max(
                                            result["emotion"].items(),
                                            key=lambda x: x[1],
                                        )[0]
                                        session_emotions[emotion] += 1

                                        # Adicionar à lista de emoções recentes para atualizar o sentimento predominante
                                        self.update_recent_emotion(emotion)

                                    if "age" in result:
                                        age = result["age"]

                            except Exception as e:
                                logger.debug(f"Falha na análise emocional: {e}")
                                pass

                        # Tentar reconhecer a face
                        face_id, face_name = None, None
                        if DEEPFACE_AVAILABLE:
                            face_id, face_name = self.recognize_face(
                                face_img, emotion, age
                            )

                        # Se não reconheceu, registrar como nova face
                        if face_id is None and DEEPFACE_AVAILABLE and face_area > 6400:
                            face_id, face_name = self.register_new_face(
                                face_img, emotion, age
                            )

                        # Atualizar informações da face na sessão atual
                        current_session_faces[face_position_key] = {
                            "face_id": face_id,
                            "face_name": face_name,
                            "last_check": time.time(),
                            "emotion": emotion,
                            "age": age,
                            "position": (x, y, w, h),
                        }

                    # Obter informações da face
                    face_info = current_session_faces.get(face_position_key, {})
                    face_id = face_info.get("face_id")
                    face_name = face_info.get("face_name", f"Face #{face_index+1}")
                    emotion = face_info.get("emotion")
                    age = face_info.get("age")

                    # Cor do retângulo - verde para faces conhecidas, azul para novas
                    rect_color = (0, 255, 0) if face_id else (255, 0, 0)

                    # Desenhar retângulo ao redor da face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

                    # Mostrar nome ou identificador
                    cv2.putText(
                        frame,
                        face_name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        rect_color,
                        2,
                    )

                    # Mostrar emoção se disponível
                    if emotion:
                        cv2.putText(
                            frame,
                            f"Sentimento: {emotion}",
                            (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )

                    # Mostrar idade se disponível
                    if age:
                        cv2.putText(
                            frame,
                            f"Idade: {age}",
                            (x, y + h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )

                # Mostrar número de faces detectadas
                cv2.putText(
                    frame,
                    f"Faces no quadro atual: {len(faces)}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Mostrar instruções
                cv2.putText(
                    frame,
                    "q: sair | s: salvar | e: estatísticas",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Exibir o frame
                cv2.imshow("Contagem de Pessoas e Análise Emocional", frame)

                # Verificar teclas pressionadas
                key = cv2.waitKey(1) & 0xFF

                # Sair se 'q' for pressionado
                if key == ord("q"):
                    logger.info("Usuário solicitou saída (tecla 'q')")
                    break

                # Salvar o frame se 's' for pressionado
                if key == ord("s"):
                    frame_path = os.path.join(
                        self.images_dir, f"webcam_frame_{frame_count}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    logger.info(f"Frame salvo em: {frame_path}")
                    print(f"Frame salvo em: {frame_path}")
                    frame_count += 1

                # Mostrar estatísticas de emoções se 'e' for pressionado
                if key == ord("e"):
                    self.show_emotion_statistics()

            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()

            # Calcular estatísticas finais
            elapsed_time = time.time() - start_time
            logger.info("=== Estatísticas de Detecção Facial ===")
            logger.info(f"Tempo total de execução: {elapsed_time:.2f} segundos")
            logger.info(f"Total de frames processados: {total_frames}")
            logger.info(f"FPS médio: {total_frames / elapsed_time:.2f}")
            logger.info(f"Total de faces detectadas: {total_faces_detected}")
            logger.info(
                f"Total de pessoas únicas identificadas: {len(self.known_faces)}"
            )

            # Mostrar estatísticas finais de emoções
            self.show_emotion_statistics()

            # Salvar resultados em JSON
            self.save_results_to_json()

            print("Detecção facial encerrada.")

        except Exception as e:
            logger.error(f"Erro na detecção facial com webcam: {e}", exc_info=True)
            return None

    def show_emotion_statistics(self):
        """Exibe estatísticas sobre as emoções detectadas e idades médias"""
        print("\n=== Estatísticas de Sentimentos Detectados ===")

        if not self.emotion_stats:
            print("Nenhum sentimento foi detectado ainda.")
            return

        # Total de detecções emocionais
        total_emotions = sum(self.emotion_stats.values())

        print(f"Total de detecções emocionais: {total_emotions}")

        # Ordenar emoções por frequência
        sorted_emotions = sorted(
            self.emotion_stats.items(), key=lambda x: x[1], reverse=True
        )

        # Mostrar contagem e percentual de cada emoção
        for emotion, count in sorted_emotions:
            percentage = (count / total_emotions) * 100
            print(f"{emotion}: {count} ocorrências ({percentage:.1f}%)")

        # Mostrar sentimento predominante e idade média por pessoa
        print("\n=== Sentimento Predominante e Idade Média por Pessoa ===")

        # Calcular idades médias e analisar emoções por pessoa
        age_averages = {}
        emotion_percentages = {}

        for face_id, face_data in self.known_faces.items():
            name = face_data["name"]
            emotions = face_data.get("emotions", {})
            ages = face_data.get("ages", [])

            # Calcular distribuição percentual de emoções para esta pessoa
            if emotions:
                total_emotion_count = sum(emotions.values())
                emotion_percentages[name] = {
                    emotion: (count / total_emotion_count) * 100
                    for emotion, count in emotions.items()
                }

                # Determinar o sentimento predominante real
                predominant = max(emotions.items(), key=lambda x: x[1])[0]

                # Log do sentimento predominante real
                logger.info(
                    f"{name}: Sentimento predominante é '{predominant}' ({emotion_percentages[name][predominant]:.1f}%)"
                )

            # Calcular média de idade
            if ages:
                avg_age = sum(ages) / len(ages)
                age_averages[name] = avg_age

                # Registrar no log
                logger.info(f"Idade média de {name}: {avg_age:.1f} anos")

        # Exibir informações de cada pessoa
        for face_id, face_data in self.known_faces.items():
            name = face_data["name"]
            emotions = face_data.get("emotions", {})
            ages = face_data.get("ages", [])

            if emotions:
                # Obter sentimento predominante real
                predominant = max(emotions.items(), key=lambda x: x[1])[0]

                # Obter percentagens de cada emoção para esta pessoa
                person_percentages = emotion_percentages.get(name, {})

                # Mostrar informações detalhadas
                print(
                    f"{name}: Sentimento predominante: {predominant} ({person_percentages.get(predominant, 0):.1f}%)"
                )
                print(f"  Distribuição de sentimentos:")

                # Mostrar cada emoção detectada e seu percentual
                for emotion, percentage in sorted(
                    person_percentages.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"    - {emotion}: {percentage:.1f}%")

                # Mostrar idade média se disponível
                if ages:
                    avg_age = sum(ages) / len(ages)
                    print(f"  Idade média: {avg_age:.1f} anos")
            else:
                if ages:
                    avg_age = sum(ages) / len(ages)
                    print(
                        f"{name}: Sem emoções detectadas (Idade média: {avg_age:.1f} anos)"
                    )
                else:
                    print(f"{name}: Sem emoções ou idade detectadas")

        # Criar e mostrar gráfico de emoções se possível
        try:
            plt.figure(figsize=(12, 12))

            # Subplot 1: Percentual de sentimentos por pessoa (anteriormente era o subplot 2)
            if emotion_percentages:
                plt.subplot(2, 1, 1)

                # Vamos criar um gráfico de barras empilhadas
                persons = list(emotion_percentages.keys())
                bottom = np.zeros(len(persons))

                # Lista de todas as possíveis emoções
                all_emotions = set()
                for person_emotions in emotion_percentages.values():
                    all_emotions.update(person_emotions.keys())
                all_emotions = sorted(all_emotions)

                # Cores para as emoções
                emotion_colors = {
                    "happy": "#99ff99",  # Verde claro
                    "sad": "#66b3ff",  # Azul claro
                    "angry": "#ff9999",  # Vermelho claro
                    "fear": "#ffcc99",  # Laranja claro
                    "disgust": "#c2c2f0",  # Lilás
                    "surprise": "#ffb3e6",  # Rosa
                    "neutral": "#e0e0e0",  # Cinza claro
                    "unknown": "#d3d3d3",  # Cinza mais claro
                }

                # Criar as barras empilhadas
                for emotion in all_emotions:
                    values = [
                        emotion_percentages[person].get(emotion, 0)
                        for person in persons
                    ]
                    color = emotion_colors.get(
                        emotion, "#000000"
                    )  # Preto como fallback
                    plt.bar(persons, values, bottom=bottom, label=emotion, color=color)

                    # Adicionar texto de percentagem em cada segmento significativo
                    for i, v in enumerate(values):
                        if v > 5:  # Mostrar apenas se o percentual for > 5%
                            plt.text(
                                i,
                                bottom[i] + v / 2,
                                f"{v:.1f}%",
                                ha="center",
                                va="center",
                            )

                    bottom += values

                plt.title("Distribuição Percentual de Sentimentos por Pessoa")
                plt.xlabel("Pessoa")
                plt.ylabel("Porcentagem (%)")
                plt.ylim(0, 100)
                plt.legend(loc="upper right")

            # Subplot 2: Idade média por pessoa (anteriormente era o subplot 3)
            if age_averages:
                plt.subplot(2, 1, 2)

                names = list(age_averages.keys())
                ages = list(age_averages.values())

                bars = plt.bar(names, ages, color="#4287f5")

                # Adicionar valor em cima de cada barra
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.5,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                    )

                plt.title("Média de Idade por Pessoa")
                plt.xlabel("Pessoas")
                plt.ylabel("Idade Média (anos)")
                plt.xticks(rotation=45)

            plt.tight_layout()

            # Salvar gráfico
            chart_path = os.path.join(
                self.images_dir,
                f"emotion_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            plt.savefig(chart_path)
            print(f"\nGráfico de sentimentos e idades salvo em: {chart_path}")

            # Mostrar gráfico
            plt.show()

        except Exception as e:
            logger.error(f"Erro ao criar gráfico: {e}")
            print("Não foi possível criar o gráfico.")

        # Registrar estatísticas no log
        logger.info("=== Estatísticas de Sentimentos ===")
        for emotion, count in sorted_emotions:
            percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
            logger.info(f"{emotion}: {count} ocorrências ({percentage:.1f}%)")

        # Registrar idades no log
        if age_averages:
            logger.info("=== Idades Médias ===")
            for name, avg_age in age_averages.items():
                logger.info(f"{name}: {avg_age:.1f} anos")

    def save_results_to_json(self):
        """Salva os resultados da detecção facial em um arquivo JSON"""
        try:
            # Preparar dados para o JSON
            timestamp = datetime.datetime.now()

            results = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "session_info": {
                    "total_detected_faces": self.total_faces_detected,
                    "unique_people": len(self.known_faces),
                    "duration_seconds": time.time() - self.start_time,
                },
                "people": [],
            }

            # Adicionar informações sobre cada pessoa
            for face_id, face_data in self.known_faces.items():
                name = face_data["name"]
                emotions = face_data.get("emotions", {})
                ages = face_data.get("ages", [])

                # Calcular emoção predominante e percentuais
                emotion_percentages = {}
                predominant_emotion = None

                if emotions:
                    total_emotion_count = sum(emotions.values())
                    emotion_percentages = {
                        emotion: (count / total_emotion_count) * 100
                        for emotion, count in emotions.items()
                    }
                    predominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

                # Calcular idade média
                avg_age = None
                if ages:
                    avg_age = sum(ages) / len(ages)

                # Adicionar dados desta pessoa
                person_data = {
                    "id": face_id,
                    "name": name,
                    "detection_count": face_data["detection_count"],
                    "first_seen": face_data["first_seen"],
                    "last_seen": face_data["last_seen"],
                    "emotions": {
                        "predominant": predominant_emotion,
                        "distribution": emotion_percentages,
                    },
                    "age": {"average": avg_age, "samples": len(ages)},
                }

                results["people"].append(person_data)

            # Gerar nome de arquivo com timestamp
            filename = f"detection_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(self.output_dir, filename)

            # Salvar arquivo JSON
            with open(file_path, "w") as f:
                json.dump(results, f, indent=4)

            logger.info(f"Resultados salvos em JSON: {file_path}")
            print(f"\nResultados salvos em JSON: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Erro ao salvar resultados em JSON: {e}")
            return None

    def list_people_count(self):
        """Lista a contagem total de pessoas identificadas e suas emoções predominantes"""
        if not self.known_faces:
            print("Nenhuma pessoa identificada ainda.")
            return

        print("\n=== Contagem de Pessoas e Sentimentos ===")
        print(f"Total de pessoas únicas identificadas: {len(self.known_faces)}")

        # Análise de sentimentos por pessoa
        emotion_by_person = {}
        age_by_person = {}

        for face_id, face_data in self.known_faces.items():
            name = face_data["name"]
            emotions = face_data.get("emotions", {})
            ages = face_data.get("ages", [])

            if emotions:
                # Pegar a emoção predominante para esta pessoa
                predominant = max(emotions.items(), key=lambda x: x[1])[0]
                emotion_by_person[name] = predominant

            if ages:
                # Calcular idade média para esta pessoa
                avg_age = sum(ages) / len(ages)
                age_by_person[name] = avg_age

        # Contar quantas pessoas têm cada emoção predominante
        emotion_count = Counter(emotion_by_person.values())

        print("\n=== Distribuição de Sentimentos ===")
        for emotion, count in emotion_count.most_common():
            percentage = (count / len(self.known_faces)) * 100
            print(f"{emotion}: {count} pessoas ({percentage:.1f}%)")

        # Estatísticas de idade (se houver)
        if age_by_person:
            print("\n=== Estatísticas de Idade ===")
            avg_ages = list(age_by_person.values())
            min_age = min(avg_ages)
            max_age = max(avg_ages)
            overall_avg = sum(avg_ages) / len(avg_ages)
            print(f"Idade média geral: {overall_avg:.1f} anos")
            print(f"Idade mínima detectada: {min_age:.1f} anos")
            print(f"Idade máxima detectada: {max_age:.1f} anos")

            # Registrar no log
            logger.info(
                f"Estatísticas de idade: Média: {overall_avg:.1f}, Min: {min_age:.1f}, Max: {max_age:.1f}"
            )

        # Listar pessoas e suas emoções e idades
        print("\n=== Pessoas Identificadas, Sentimentos e Idades ===")
        for face_id, face_data in self.known_faces.items():
            name = face_data["name"]
            emotions = face_data.get("emotions", {})
            ages = face_data.get("ages", [])
            detection_count = face_data["detection_count"]

            if emotions:
                predominant = max(emotions.items(), key=lambda x: x[1])[0]
                if ages:
                    avg_age = sum(ages) / len(ages)
                    print(
                        f"{name} - Sentimento predominante: {predominant} (Idade média: {avg_age:.1f} anos, Detectado {detection_count} vezes)"
                    )
                else:
                    print(
                        f"{name} - Sentimento predominante: {predominant} (Idade não detectada, Detectado {detection_count} vezes)"
                    )
            else:
                if ages:
                    avg_age = sum(ages) / len(ages)
                    print(
                        f"{name} - Sem emoções detectadas (Idade média: {avg_age:.1f} anos, Detectado {detection_count} vezes)"
                    )
                else:
                    print(
                        f"{name} - Sem emoções ou idade detectadas (Detectado {detection_count} vezes)"
                    )


def main():
    """Função principal para demonstrar o uso da classe FaceCounter"""
    logger.info("=== Iniciando aplicação Face Counter & Emotion Analyzer ===")

    counter = FaceCounter()

    print("=" * 50)
    print("Contador de Pessoas & Analisador de Sentimentos")
    print("=" * 50)
    logger.info(
        f"Status do DeepFace: {'Disponível' if DEEPFACE_AVAILABLE else 'Indisponível'}"
    )

    # Menu de opções
    while True:
        print("\nOpções:")
        print("1. Detectar e contar pessoas em tempo real (webcam)")
        print("2. Ver estatísticas de sentimentos")
        print("3. Ver contagem de pessoas identificadas")
        print("4. Salvar resultados em JSON")
        print("5. Sair")

        choice = input("\nEscolha uma opção (1-5): ")
        logger.info(f"Opção selecionada: {choice}")

        if choice == "1":
            counter.detect_faces_webcam()

        elif choice == "2":
            counter.show_emotion_statistics()

        elif choice == "3":
            counter.list_people_count()

        elif choice == "4":
            file_path = counter.save_results_to_json()
            if file_path:
                print(f"Resultados salvos em: {file_path}")

        elif choice == "5":
            logger.info("Usuário solicitou saída do programa")
            print("\nSaindo do programa. Até mais!")
            break

        else:
            logger.warning(f"Opção inválida escolhida: {choice}")
            print("\nOpção inválida. Por favor, escolha uma opção de 1 a 5.")

    logger.info("=== Aplicação Face Counter & Emotion Analyzer encerrada ===")


if __name__ == "__main__":
    main()
