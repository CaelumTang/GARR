import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def l2norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    require(np.all(np.isfinite(x)), "weights contain NaN/Inf")
    exp_x = np.exp(x - float(np.max(x)))
    denom = float(np.sum(exp_x))
    require(denom > 0.0 and math.isfinite(denom), "softmax denominator must be finite and > 0")
    return (exp_x / denom).astype(np.float32)


@dataclass(frozen=True)
class Bundle:
    ids: np.ndarray
    v: np.ndarray
    t: np.ndarray
    pre: np.ndarray
    y: np.ndarray
    id_to_row: dict[int, int]


def load_bundle(npz_path: str) -> Bundle:
    npz_path = os.path.abspath(npz_path)
    require(os.path.isfile(npz_path), f"npz not found: {npz_path}")
    data = np.load(npz_path)

    required_keys = {"video_id", "vision_emb", "text_emb", "ground_truth", "pre_score"}
    missing_keys = sorted(required_keys - set(data.keys()))
    require(len(missing_keys) == 0, f"{npz_path}: missing keys={missing_keys}")

    ids = np.asarray(data["video_id"], dtype=np.int64)
    vision = np.asarray(data["vision_emb"], dtype=np.float32)
    text = np.asarray(data["text_emb"], dtype=np.float32)
    target = np.asarray(data["ground_truth"], dtype=np.float32)
    pre_score = np.asarray(data["pre_score"], dtype=np.float32)

    require(ids.ndim == 1, f"{npz_path}: video_id must be 1-D")
    require(vision.ndim == 2 and text.ndim == 2, f"{npz_path}: vision/text must be 2-D")
    require(target.ndim == 1 and target.shape[0] == ids.shape[0], f"{npz_path}: ground_truth must align with ids")
    require(pre_score.ndim == 1 and pre_score.shape[0] == ids.shape[0], f"{npz_path}: pre_score must align with ids")
    require(vision.shape[0] == text.shape[0] == ids.shape[0], f"{npz_path}: embedding rows mismatch with ids")
    require(vision.shape[1] == text.shape[1], f"{npz_path}: vision/text dim mismatch")
    require(np.all(np.isfinite(vision)), f"{npz_path}: vision_emb contains NaN/Inf")
    require(np.all(np.isfinite(text)), f"{npz_path}: text_emb contains NaN/Inf")
    require(np.all(np.isfinite(target)), f"{npz_path}: ground_truth contains NaN/Inf")
    require(np.all(np.isfinite(pre_score)), f"{npz_path}: pre_score contains NaN/Inf")

    vision = l2norm(vision).astype(np.float32)
    text = l2norm(text).astype(np.float32)
    id_to_row = {int(video_id): int(row) for row, video_id in enumerate(ids.tolist())}
    return Bundle(ids=ids, v=vision, t=text, pre=pre_score, y=target, id_to_row=id_to_row)


def parse_neighbors_csv(path: str, *, k: int) -> dict[int, tuple[list[int], list[float]]]:
    path = os.path.abspath(path)
    require(os.path.isfile(path), f"neighbors csv not found: {path}")
    parsed: dict[int, tuple[list[int], list[float]]] = {}
    with open(path, "r", encoding="utf-8") as file:
        header = file.readline()
        require(header.strip().startswith("video_id"), f"Invalid neighbors header in {path}: {header.strip()}")
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            require(len(parts) == 3, f"Invalid neighbors row: {line[:120]}")
            query_id = int(parts[0])
            neighbor_ids = [int(item) for item in parts[1].strip().split() if item][: int(k)]
            neighbor_sims = [float(item) for item in parts[2].strip().split() if item][: int(k)]
            require(len(neighbor_ids) == len(neighbor_sims) and len(neighbor_ids) > 0, f"Invalid neighbors for qid={query_id}")
            parsed[query_id] = (neighbor_ids, neighbor_sims)
    require(len(parsed) > 0, f"No rows parsed from neighbors csv: {path}")
    return parsed


class Stage3Dataset(Dataset):
    def __init__(
        self,
        *,
        train_npz: str,
        val_npz: str,
        test_npz: str,
        neighbors_csv: str,
        split: str,
        k: int,
    ):
        split = str(split).lower().strip()
        require(split in {"train", "val", "test"}, f"Invalid split: {split}")
        self.split = split
        self.k = int(k)
        require(self.k > 0, "--k must be > 0")

        self.tr = load_bundle(train_npz)
        self.va = load_bundle(val_npz)
        self.te = load_bundle(test_npz)
        self.q = {"train": self.tr, "val": self.va, "test": self.te}[self.split]

        self.id_to_y: dict[int, float] = {}
        self.id_to_pre: dict[int, float] = {}
        self.id_to_v: dict[int, np.ndarray] = {}
        self.id_to_t: dict[int, np.ndarray] = {}
        for bundle in (self.tr, self.va, self.te):
            for video_id, row in bundle.id_to_row.items():
                self.id_to_y[int(video_id)] = float(bundle.y[row])
                self.id_to_pre[int(video_id)] = float(bundle.pre[row])
                self.id_to_v[int(video_id)] = bundle.v[row]
                self.id_to_t[int(video_id)] = bundle.t[row]

        neighbor_map = parse_neighbors_csv(neighbors_csv, k=self.k)
        train_ids = set(self.tr.id_to_row.keys())
        allowed_neighbors = train_ids if self.split in {"train", "val"} else train_ids | set(self.va.id_to_row.keys())

        self.samples: list[tuple[int, list[int], np.ndarray]] = []
        for query_id in self.q.ids.tolist():
            query_id = int(query_id)
            if query_id not in neighbor_map:
                continue
            neighbor_ids, neighbor_sims = neighbor_map[query_id]
            require(len(neighbor_ids) == self.k and len(neighbor_sims) == self.k, f"Neighbors length mismatch for qid={query_id}")
            for neighbor_id in neighbor_ids:
                require(int(neighbor_id) in allowed_neighbors, f"Neighbor id {neighbor_id} not allowed for split={self.split}")
            self.samples.append((query_id, neighbor_ids, softmax(np.asarray(neighbor_sims, dtype=np.float64))))

        require(len(self.samples) > 0, f"No samples found for split={self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        query_id, neighbor_ids, weights = self.samples[index]
        neighbor_v = np.stack([self.id_to_v[int(item)].astype(np.float32) for item in neighbor_ids], axis=0)
        neighbor_t = np.stack([self.id_to_t[int(item)].astype(np.float32) for item in neighbor_ids], axis=0)
        neighbor_y = np.asarray([self.id_to_y[int(item)] for item in neighbor_ids], dtype=np.float32).reshape(self.k, 1)
        return {
            "qid": int(query_id),
            "q_v": torch.from_numpy(self.id_to_v[query_id].astype(np.float32)),
            "q_t": torch.from_numpy(self.id_to_t[query_id].astype(np.float32)),
            "nb_v": torch.from_numpy(neighbor_v),
            "nb_t": torch.from_numpy(neighbor_t),
            "w": torch.from_numpy(weights.astype(np.float32)),
            "nb_y": torch.from_numpy(neighbor_y),
            "q_pre": torch.tensor([float(self.id_to_pre[query_id])], dtype=torch.float32),
            "y": torch.tensor([float(self.id_to_y[query_id])], dtype=torch.float32),
        }
