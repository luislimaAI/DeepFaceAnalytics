"""Main application module orchestrating all sub-modules."""

import datetime
import json
import logging
import os
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from deepface_analytics.analyzer import EMOTION_RECHECK_INTERVAL, FaceAnalyzer
from deepface_analytics.detector import FaceDetector
from deepface_analytics.storage import FaceStorage
from deepface_analytics.tracker import FaceTracker

try:
    from deepface import DeepFace as _DeepFace

    DEEPFACE_AVAILABLE: bool = True
except ImportError:
    DEEPFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Webcam configuration
WEBCAM_WIDTH: int = 640
WEBCAM_HEIGHT: int = 480
WEBCAM_FPS: int = 30
FACE_AREA_MIN: int = 3600
FACE_AREA_MIN_REGISTER: int = 6400
FACE_WIDTH_RATIO_MAX: float = 0.7
CLEANUP_INTERVAL: float = 5.0


class FaceCounterApp:
    """Orchestrates FaceDetector, FaceAnalyzer, FaceTracker, and FaceStorage."""

    def __init__(self, no_deepface: bool = False) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(base_dir)
        self.known_faces_dir = os.path.join(project_dir, "images", "known_faces")
        self.images_dir = os.path.join(project_dir, "images")
        self.output_dir = os.path.join(project_dir, "output")
        os.makedirs(self.known_faces_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.no_deepface = no_deepface

        self.detector = FaceDetector()
        self.analyzer = FaceAnalyzer()
        self.tracker = FaceTracker()
        self.storage = FaceStorage()

        faces_json = os.path.join(self.known_faces_dir, "known_faces.json")
        self.storage.known_faces = self.storage.load_known_faces(faces_json)

        self.start_time: float = time.time()
        self.total_faces_detected: int = 0
        self.emotion_stats: Dict[str, int] = {}

    def warmup_models(self) -> None:
        """Pre-warm DeepFace models before the main capture loop."""
        if not DEEPFACE_AVAILABLE or self.no_deepface:
            return
        print("Carregando modelos de análise... (pode levar alguns segundos)")
        t0 = time.time()
        try:
            blank: npt.NDArray[Any] = np.zeros((224, 224, 3), dtype=np.uint8)
            _DeepFace.analyze(
                blank,
                actions=["emotion", "age", "embedding"],
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            logger.exception("Warmup failed (non-fatal)")
        elapsed = time.time() - t0
        logger.info("Modelos carregados em %.2f segundos", elapsed)

    def detect_faces_webcam(self) -> None:
        """Main webcam capture and display loop using the new modules."""
        logger.info("Iniciando detecção facial pela webcam com análise emocional")

        self.warmup_models()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error("Erro: Não foi possível acessar a webcam")
            return

        logger.info("Aguardando inicialização da câmera...")
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                time.sleep(0.1)

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            logger.error("Erro: Não foi possível obter frames da câmera após inicialização")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Webcam iniciada com sucesso. Resolução: %dx%d", width, height)
        print("Iniciando detecção facial com contagem de pessoas e análise emocional...")
        print(
            "Pressione 'q' para sair, 's' para salvar um frame,"
            " 'e' para ver estatísticas de emoções"
        )

        frame_count = 0
        total_frames = 0
        total_faces_session = 0
        start_time = time.time()
        fps_window: Deque[float] = deque(maxlen=30)
        fps = 0.0
        can_analyze = DEEPFACE_AVAILABLE and not self.no_deepface
        logger.info("Análise emocional: %s", "ATIVA" if can_analyze else "INATIVA")

        current_session_faces: Dict[str, Any] = {}
        last_cleanup = time.time()
        faces_json = os.path.join(self.known_faces_dir, "known_faces.json")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Erro: Falha ao capturar frame da webcam")
                break

            total_frames += 1
            now = time.time()
            fps_window.append(now)
            if len(fps_window) >= 2:
                fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])

            faces = self.detector.detect_faces(frame)
            total_faces_session += len(faces)
            self.total_faces_detected += len(faces)

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"Pessoas únicas identificadas: {len(self.storage.known_faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if self.emotion_stats:
                top_emotion = max(self.emotion_stats, key=lambda e: self.emotion_stats[e])
                cv2.putText(
                    frame,
                    f"Sentimento predominante: {top_emotion}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            for face_index, (x, y, w, h) in enumerate(faces):
                face_area = w * h
                face_width_ratio = w / frame.shape[1]
                if face_area < FACE_AREA_MIN or face_width_ratio > FACE_WIDTH_RATIO_MAX:
                    continue

                face_img: npt.NDArray[Any] = frame[y : y + h, x : x + w]
                face_position_key = f"{x}_{y}_{w}_{h}"

                last_check = float(
                    current_session_faces.get(face_position_key, {}).get("last_check", 0)
                )
                if can_analyze and (
                    face_position_key not in current_session_faces
                    or now - last_check > EMOTION_RECHECK_INTERVAL
                ):
                    emotion: Optional[str] = None
                    age: Optional[int] = None
                    face_id: Optional[str] = None
                    face_name: Optional[str] = None

                    result = self.analyzer.analyze_face(face_img, face_position_key)
                    if result is not None:
                        raw_emotion = result.get("dominant_emotion", "")
                        emotion = str(raw_emotion) if raw_emotion else None
                        raw_age = result.get("age", 0)
                        age = int(raw_age) if raw_age else None

                        embedding: List[Any] = result.get("embedding", [])
                        if embedding:
                            enc: npt.NDArray[Any] = np.array(embedding)
                            face_id = self.tracker.is_same_as_known_face(enc)

                            if face_id is None and face_area > FACE_AREA_MIN_REGISTER:
                                face_id, face_name = self.storage.register_face(
                                    face_img, emotion, age, self.known_faces_dir
                                )
                                self.tracker.add_face_encoding(face_id, enc)
                                self.storage.save_known_faces(
                                    self.storage.known_faces, faces_json
                                )
                            elif face_id and face_id in self.storage.known_faces:
                                fd: Dict[str, Any] = self.storage.known_faces[face_id]
                                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                fd["last_seen"] = now_str
                                fd["detection_count"] = int(fd.get("detection_count", 0)) + 1
                                if emotion:
                                    emo_dict: Dict[str, int] = fd.setdefault("emotions", {})
                                    emo_dict[emotion] = emo_dict.get(emotion, 0) + 1
                                if age is not None:
                                    ages_list: List[Any] = fd.setdefault("ages", [])
                                    ages_list.append(age)

                        if emotion:
                            self.emotion_stats[emotion] = self.emotion_stats.get(emotion, 0) + 1
                            self.tracker.update_recent_emotion(emotion)

                        if face_id and face_name is None and face_id in self.storage.known_faces:
                            face_name = str(
                                self.storage.known_faces[face_id].get("name", "")
                            )

                    current_session_faces[face_position_key] = {
                        "face_id": face_id,
                        "face_name": face_name,
                        "last_check": now,
                        "emotion": emotion,
                        "age": age,
                        "position": (x, y, w, h),
                    }

                face_info = current_session_faces.get(face_position_key, {})
                disp_face_id: Optional[str] = face_info.get("face_id")
                disp_face_name: str = str(
                    face_info.get("face_name") or f"Face #{face_index + 1}"
                )
                disp_emotion: Optional[str] = (
                    "N/A" if self.no_deepface else face_info.get("emotion")
                )
                disp_age: Optional[int] = face_info.get("age")

                rect_color = (0, 255, 0) if disp_face_id else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
                cv2.putText(
                    frame,
                    disp_face_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    rect_color,
                    2,
                )
                if disp_emotion:
                    cv2.putText(
                        frame,
                        f"Sentimento: {disp_emotion}",
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                if disp_age:
                    cv2.putText(
                        frame,
                        f"Idade: {disp_age}",
                        (x, y + h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

            cv2.putText(
                frame,
                f"Faces no quadro atual: {len(faces)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "q: sair | s: salvar | e: estatísticas",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Contagem de Pessoas e Análise Emocional", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Usuário solicitou saída (tecla 'q')")
                break
            if key == ord("s"):
                frame_path = os.path.join(
                    self.images_dir, f"webcam_frame_{frame_count}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                logger.info("Frame salvo em: %s", frame_path)
                print(f"Frame salvo em: {frame_path}")
                frame_count += 1
            if key == ord("e"):
                self.show_emotion_statistics()

            if now - last_cleanup > CLEANUP_INTERVAL:
                self.tracker.cleanup_trackers(current_session_faces)
                last_cleanup = now

        cap.release()
        cv2.destroyAllWindows()

        elapsed_time = time.time() - start_time
        logger.info("=== Estatísticas de Detecção Facial ===")
        logger.info("Tempo total de execução: %.2f segundos", elapsed_time)
        logger.info("Total de frames processados: %d", total_frames)
        if elapsed_time > 0:
            logger.info("FPS médio: %.2f", total_frames / elapsed_time)
        logger.info("Total de faces detectadas: %d", total_faces_session)
        logger.info(
            "Total de pessoas únicas identificadas: %d", len(self.storage.known_faces)
        )
        self.show_emotion_statistics()
        self.save_results_to_json()
        print("Detecção facial encerrada.")

    def show_emotion_statistics(self) -> None:
        """Display emotion statistics (identical user-facing output to legacy)."""
        print("\n=== Estatísticas de Sentimentos Detectados ===")

        if not self.emotion_stats:
            print("Nenhum sentimento foi detectado ainda.")
            return

        total_emotions = sum(self.emotion_stats.values())
        print(f"Total de detecções emocionais: {total_emotions}")

        sorted_emotions = sorted(
            self.emotion_stats.items(), key=lambda x: x[1], reverse=True
        )
        for emotion, count in sorted_emotions:
            percentage = (count / total_emotions) * 100
            print(f"{emotion}: {count} ocorrências ({percentage:.1f}%)")

        print("\n=== Sentimento Predominante e Idade Média por Pessoa ===")

        age_averages: Dict[str, float] = {}
        emotion_percentages: Dict[str, Dict[str, float]] = {}

        for face_id, face_data in self.storage.known_faces.items():
            name: str = str(face_data.get("name", face_id))
            emotions: Dict[str, int] = face_data.get("emotions", {})
            ages: List[Any] = face_data.get("ages", [])

            if emotions:
                total_count = sum(emotions.values())
                person_pct = {em: (cnt / total_count) * 100 for em, cnt in emotions.items()}
                emotion_percentages[name] = person_pct
                predominant = max(emotions, key=lambda e: emotions[e])
                logger.info(
                    "%s: Sentimento predominante é '%s' (%.1f%%)",
                    name,
                    predominant,
                    person_pct[predominant],
                )
                print(
                    f"{name}: Sentimento predominante: {predominant}"
                    f" ({person_pct[predominant]:.1f}%)"
                )
                print("  Distribuição de sentimentos:")
                for em, pct in sorted(
                    person_pct.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"    - {em}: {pct:.1f}%")
            else:
                print(f"{name}: Sem emoções detectadas")

            if ages:
                avg_age = sum(float(a) for a in ages) / len(ages)
                age_averages[name] = avg_age
                logger.info("Idade média de %s: %.1f anos", name, avg_age)
                print(f"  Idade média: {avg_age:.1f} anos")

        try:
            plt.figure(figsize=(12, 12))

            if emotion_percentages:
                plt.subplot(2, 1, 1)
                persons = list(emotion_percentages.keys())
                bottom: npt.NDArray[Any] = np.zeros(len(persons))
                all_emotions: set[str] = set()
                for person_pcts in emotion_percentages.values():
                    all_emotions.update(person_pcts.keys())
                emotion_colors: Dict[str, str] = {
                    "happy": "#99ff99",
                    "sad": "#66b3ff",
                    "angry": "#ff9999",
                    "fear": "#ffcc99",
                    "disgust": "#c2c2f0",
                    "surprise": "#ffb3e6",
                    "neutral": "#e0e0e0",
                    "unknown": "#d3d3d3",
                }
                for emo in sorted(all_emotions):
                    values = [emotion_percentages[p].get(emo, 0) for p in persons]
                    color = emotion_colors.get(emo, "#000000")
                    plt.bar(persons, values, bottom=bottom, label=emo, color=color)
                    for i, v in enumerate(values):
                        if v > 5:
                            plt.text(
                                i,
                                float(bottom[i]) + v / 2,
                                f"{v:.1f}%",
                                ha="center",
                                va="center",
                            )
                    bottom = bottom + np.array(values)
                plt.title("Distribuição Percentual de Sentimentos por Pessoa")
                plt.xlabel("Pessoa")
                plt.ylabel("Porcentagem (%)")
                plt.ylim(0, 100)
                plt.legend(loc="upper right")

            if age_averages:
                plt.subplot(2, 1, 2)
                names = list(age_averages.keys())
                ages_vals = list(age_averages.values())
                bars = plt.bar(names, ages_vals, color="#4287f5")
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
            chart_path = os.path.join(
                self.images_dir,
                f"emotion_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            plt.savefig(chart_path)
            print(f"\nGráfico de sentimentos e idades salvo em: {chart_path}")
            plt.show()
        except Exception:
            logger.exception("Erro ao criar gráfico")
            print("Não foi possível criar o gráfico.")

        logger.info("=== Estatísticas de Sentimentos ===")
        for emotion, count in sorted_emotions:
            percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
            logger.info("%s: %d ocorrências (%.1f%%)", emotion, count, percentage)

        if age_averages:
            logger.info("=== Idades Médias ===")
            for name, avg_age in age_averages.items():
                logger.info("%s: %.1f anos", name, avg_age)

    def list_people_count(self) -> None:
        """Show people count statistics (identical user-facing output to legacy)."""
        if not self.storage.known_faces:
            print("Nenhuma pessoa identificada ainda.")
            return

        print("\n=== Contagem de Pessoas e Sentimentos ===")
        print(f"Total de pessoas únicas identificadas: {len(self.storage.known_faces)}")

        emotion_by_person: Dict[str, str] = {}
        age_by_person: Dict[str, float] = {}

        for face_id, face_data in self.storage.known_faces.items():
            name: str = str(face_data.get("name", face_id))
            emotions: Dict[str, int] = face_data.get("emotions", {})
            ages: List[Any] = face_data.get("ages", [])

            if emotions:
                predominant = max(emotions, key=lambda e: emotions[e])
                emotion_by_person[name] = predominant
            if ages:
                age_by_person[name] = sum(float(a) for a in ages) / len(ages)

        emotion_count: Counter[str] = Counter(emotion_by_person.values())
        print("\n=== Distribuição de Sentimentos ===")
        for emotion, count in emotion_count.most_common():
            percentage = (count / len(self.storage.known_faces)) * 100
            print(f"{emotion}: {count} pessoas ({percentage:.1f}%)")

        if age_by_person:
            print("\n=== Estatísticas de Idade ===")
            avg_ages_list = list(age_by_person.values())
            min_age = min(avg_ages_list)
            max_age = max(avg_ages_list)
            overall_avg = sum(avg_ages_list) / len(avg_ages_list)
            print(f"Idade média geral: {overall_avg:.1f} anos")
            print(f"Idade mínima detectada: {min_age:.1f} anos")
            print(f"Idade máxima detectada: {max_age:.1f} anos")
            logger.info(
                "Estatísticas de idade: Média: %.1f, Min: %.1f, Max: %.1f",
                overall_avg,
                min_age,
                max_age,
            )

        print("\n=== Pessoas Identificadas, Sentimentos e Idades ===")
        for face_id, face_data in self.storage.known_faces.items():
            name = str(face_data.get("name", face_id))
            emotions = face_data.get("emotions", {})
            ages = face_data.get("ages", [])
            detection_count = int(face_data.get("detection_count", 0))
            avg_age_str = (
                f"Idade média: {sum(float(a) for a in ages) / len(ages):.1f} anos"
                if ages
                else "Idade não detectada"
            )
            if emotions:
                predominant = max(emotions, key=lambda e: emotions[e])
                print(
                    f"{name} - Sentimento predominante: {predominant}"
                    f" ({avg_age_str}, Detectado {detection_count} vezes)"
                )
            else:
                print(
                    f"{name} - Sem emoções detectadas"
                    f" ({avg_age_str}, Detectado {detection_count} vezes)"
                )

    def save_results_to_json(self) -> Optional[str]:
        """Save session results to JSON using FaceStorage.generate_session_json()."""
        try:
            stats: Dict[str, Any] = {"total_detected_faces": self.total_faces_detected}
            results = self.storage.generate_session_json(
                self.storage.known_faces, stats, self.start_time
            )
            timestamp = datetime.datetime.now()
            filename = f"detection_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(self.output_dir, filename)
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=4)
            logger.info("Resultados salvos em JSON: %s", file_path)
            print(f"\nResultados salvos em JSON: {file_path}")
            return file_path
        except Exception:
            logger.exception("Erro ao salvar resultados em JSON")
            return None


def main(no_deepface: bool = False) -> None:
    """Main entry point for the Face Counter & Emotion Analyzer application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(os.path.dirname(base_dir), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"face_counter_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info("=== Iniciando aplicação Face Counter & Emotion Analyzer ===")

    app = FaceCounterApp(no_deepface=no_deepface)

    print("=" * 50)
    print("Contador de Pessoas & Analisador de Sentimentos")
    print("=" * 50)
    logger.info(
        "Status do DeepFace: %s", "Disponível" if DEEPFACE_AVAILABLE else "Indisponível"
    )

    while True:
        print("\nOpções:")
        print("1. Detectar e contar pessoas em tempo real (webcam)")
        print("2. Ver estatísticas de sentimentos")
        print("3. Ver contagem de pessoas identificadas")
        print("4. Salvar resultados em JSON")
        print("5. Sair")

        choice = input("\nEscolha uma opção (1-5): ").strip()
        logger.info("Opção selecionada: %s", choice)

        if choice not in {"1", "2", "3", "4", "5"}:
            logger.warning("Opção inválida escolhida: %r", choice)
            print("\nOpção inválida. Por favor, escolha uma opção de 1 a 5.")
            continue

        if choice == "1":
            app.detect_faces_webcam()
        elif choice == "2":
            app.show_emotion_statistics()
        elif choice == "3":
            app.list_people_count()
        elif choice == "4":
            file_path = app.save_results_to_json()
            if file_path:
                print(f"Resultados salvos em: {file_path}")
        elif choice == "5":
            logger.info("Usuário solicitou saída do programa")
            print("\nSaindo do programa. Até mais!")
            break

    logger.info("=== Aplicação Face Counter & Emotion Analyzer encerrada ===")
