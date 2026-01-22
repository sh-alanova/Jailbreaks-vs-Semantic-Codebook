import json
import os
import urllib.request
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CODEBOOK_DIR = os.getenv(
    "CODEBOOK_DIR", os.path.join(os.path.dirname(__file__), "codebook")
)


def ensure_codebook_exists(
    local_path: str = "codebook/codebook_embeddings.jsonl",
    url: str = "https://huggingface.co/datasets/shalanova/codebook_embeddings/resolve/main/codebook_embeddings.jsonl",
):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Скачивание кодбука из {url}...")
        urllib.request.urlretrieve(url, local_path)


def load_embs(save_file: str) -> List[Dict[str, Any]]:
    """
    Загружает данные из файла в формате JSON Lines.
    """

    data = []
    with open(save_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if len(row["text"].split()) == 1:
                    print(row["text"])
                else:
                    data.append(row)
    return data


class SemanticJailbreakFilter:
    def __init__(
        self,
        codebook_path: str = None,
        model: SentenceTransformer = None,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        threshold: float = 0.7,
    ):
        """
        Инициализирует семантический фильтр для обнаружения мультиязычных jailbreak-запросов
        """

        self.threshold = threshold
        print("Начинаем загружать модель...")
        if not model:
            self.model = SentenceTransformer(model_name, device=device)
            self.model.eval()
        else:
            self.model = model

        if codebook_path is None:
            codebook_path = os.path.join(CODEBOOK_DIR, "codebook_embeddings.jsonl")

        if not os.path.exists(codebook_path):
            ensure_codebook_exists(codebook_path)

        df = load_embs(codebook_path)
        embeddings = [x["embedding"] for x in df]
        self.codebook = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # print(f"Кодбук загружен, в нём {self.codebook.shape[0]} векторов")

    def _is_harmful(self, prompt: str, threshold: float = None) -> bool:
        """
        Определяет, является ли запрос потенциально вредоносным.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            return False

        thr = threshold if threshold is not None else self.threshold
        emb = self.model.encode([prompt], normalize_embeddings=True)[0]
        emb = emb.reshape(1, -1)

        sims = cosine_similarity(emb, self.codebook)[0]
        max_sim = float(sims.max())

        return max_sim >= thr


_filter_instance = None  # lazy init


def is_harmful(
    prompt: str, model: SentenceTransformer = None, threshold: float = 0.7
) -> bool:
    """
    Функция-обёртка для быстрого использования.
    Инициализирует фильтр при первом вызове.
    """

    global _filter_instance

    if _filter_instance is None:
        print("Первый вызов --> Инициализируем фильтр...")
        _filter_instance = SemanticJailbreakFilter(model=model, threshold=threshold)
        print("Инициализация закончена.")

    return _filter_instance._is_harmful(prompt, threshold=threshold)
