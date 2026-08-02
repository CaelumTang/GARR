from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _softmax(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float64)
    _require(np.all(np.isfinite(w)), "weights contain NaN/Inf")
    m = float(np.max(w))
    ex = np.exp(w - m)
    s = float(np.sum(ex))
    _require(s > 0.0 and np.isfinite(s), "softmax denominator must be finite and > 0")
    return (ex / s).astype(np.float32)


@dataclass(frozen=True)
class Bundle:
    ids: np.ndarray
    v: np.ndarray
    t: np.ndarray
    pre: np.ndarray
    y: np.ndarray
    id_to_row: dict[int, int]


def load_bundle(npz_path: str) -> Bundle:
    p = os.path.abspath(npz_path)
    _require(os.path.isfile(p), f"npz not found: {p}")
    with np.load(p) as data:
        keys = set(data.keys())
        required = {"video_id", "vision_emb", "text_emb", "pre_score", "ground_truth"}
        missing = sorted(required - keys)
        _require(not missing, f"{p}: missing keys={missing}, got keys={sorted(keys)}")

        ids = np.asarray(data["video_id"], dtype=np.int64)
        v = np.asarray(data["vision_emb"], dtype=np.float32)
        t = np.asarray(data["text_emb"], dtype=np.float32)
        y = np.asarray(data["ground_truth"], dtype=np.float32)
        pre = np.asarray(data["pre_score"], dtype=np.float32)

    _require(ids.ndim == 1, f"{p}: video_id must be 1-D, got {ids.shape}")
    _require(
        v.ndim == 2 and t.ndim == 2, f"{p}: vision/text must be 2-D, got v={v.shape} t={t.shape}"
    )
    _require(
        y.ndim == 1 and y.shape[0] == ids.shape[0],
        f"{p}: ground_truth must be 1-D aligned with ids",
    )
    _require(v.shape[0] == ids.shape[0] == t.shape[0], f"{p}: embedding rows mismatch with ids")
    _require(
        v.shape[1] == t.shape[1], f"{p}: vision/text dim mismatch v={v.shape[1]} t={t.shape[1]}"
    )
    _require(len(set(ids.tolist())) == ids.size, f"{p}: video_id values must be unique")
    _require(np.all(np.isfinite(v)), f"{p}: vision_emb contains NaN/Inf")
    _require(np.all(np.isfinite(t)), f"{p}: text_emb contains NaN/Inf")
    _require(np.all(np.isfinite(y)), f"{p}: ground_truth contains NaN/Inf")

    _require(pre.ndim == 1 and pre.shape[0] == ids.shape[0], f"{p}: pre_score must align")
    _require(
        np.all(np.isfinite(pre)),
        f"{p}: pre_score contains NaN/Inf; inspect the MLLM-generated text",
    )

    v = _l2norm(v).astype(np.float32)
    t = _l2norm(t).astype(np.float32)

    id_to_row = {int(i): int(r) for r, i in enumerate(ids.tolist())}
    return Bundle(ids=ids, v=v, t=t, pre=pre, y=y, id_to_row=id_to_row)


def _parse_neighbors_csv(path: str, *, k: int) -> dict[int, tuple[list[int], list[float]]]:
    p = os.path.abspath(path)
    _require(os.path.isfile(p), f"neighbors csv not found: {p}")
    out: dict[int, tuple[list[int], list[float]]] = {}
    with open(p, encoding="utf-8") as f:
        header = f.readline()
        _require(
            header.strip().startswith("video_id"),
            f"Invalid neighbors header in {p}: {header.strip()}",
        )
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            _require(len(parts) == 3, f"Invalid neighbors row (expect 3 columns): {line[:120]}")
            qid = int(parts[0])
            nb_ids = [int(x) for x in parts[1].strip().split() if x][: int(k)]
            nb_sims = [float(x) for x in parts[2].strip().split() if x][: int(k)]
            _require(len(nb_ids) > 0 and len(nb_sims) > 0, f"Empty neighbors for qid={qid} in {p}")
            _require(
                len(nb_ids) == len(nb_sims), f"Neighbor id/sim length mismatch for qid={qid} in {p}"
            )
            _require(qid not in out, f"Duplicate query video_id={qid} in {p}")
            _require(len(set(nb_ids)) == len(nb_ids), f"Duplicate neighbor for qid={qid} in {p}")
            out[qid] = (nb_ids, nb_sims)
    _require(len(out) > 0, f"No rows parsed from neighbors csv: {p}")
    return out


class PredictorDataset(Dataset):
    """Retrieval Refinement dataset."""

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
        _require(split in {"train", "val", "test"}, f"Invalid split: {split}")
        self.split = split
        self.k = int(k)
        _require(self.k > 0, "--k must be > 0")

        self.tr = load_bundle(train_npz)
        self.va = load_bundle(val_npz)
        self.te = load_bundle(test_npz)

        self.q = {"train": self.tr, "val": self.va, "test": self.te}[self.split]
        train_ids = set(self.tr.id_to_row)
        val_ids = set(self.va.id_to_row)
        test_ids = set(self.te.id_to_row)
        _require(not (train_ids & val_ids), "train and val video IDs overlap")
        _require(not (train_ids & test_ids), "train and test video IDs overlap")
        _require(not (val_ids & test_ids), "val and test video IDs overlap")

        self._id_to_y: dict[int, float] = {}
        self._id_to_pre: dict[int, float] = {}
        self._id_to_v: dict[int, np.ndarray] = {}
        self._id_to_t: dict[int, np.ndarray] = {}
        for b in (self.tr, self.va, self.te):
            for vid, row in b.id_to_row.items():
                self._id_to_y[int(vid)] = float(b.y[row])
                self._id_to_pre[int(vid)] = float(b.pre[row])
                self._id_to_v[int(vid)] = b.v[row]
                self._id_to_t[int(vid)] = b.t[row]

        nb_map = _parse_neighbors_csv(neighbors_csv, k=self.k)
        trainval_ids = train_ids | val_ids
        allowed_neighbors = train_ids if self.split in {"train", "val"} else trainval_ids

        self.samples: list[tuple[int, list[int], np.ndarray]] = []
        for qid_raw in self.q.ids.tolist():
            qid = int(qid_raw)
            _require(qid in nb_map, f"Missing neighbors for query video_id={qid}")
            neighbor_ids, similarities = nb_map[qid]
            _require(
                len(neighbor_ids) == self.k and len(similarities) == self.k,
                f"Expected {self.k} neighbors for video_id={qid}, got {len(neighbor_ids)}",
            )
            if self.split == "train":
                _require(qid not in neighbor_ids, f"Train query {qid} retrieves itself")
            for neighbor_id in neighbor_ids:
                _require(
                    int(neighbor_id) in allowed_neighbors,
                    f"Neighbor {neighbor_id} is not allowed for split={self.split}",
                )
            weights = _softmax(np.asarray(similarities, dtype=np.float64))
            self.samples.append((qid, neighbor_ids, weights))

        _require(
            len(self.samples) > 0,
            f"No samples matched for split={self.split} from neighbors_csv={neighbors_csv}",
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        qid, nb_ids, w = self.samples[idx]

        q_v = self._id_to_v[qid].astype(np.float32)
        q_t = self._id_to_t[qid].astype(np.float32)
        y = float(self._id_to_y[qid])

        out = {
            "qid": int(qid),
            "q_v": torch.from_numpy(q_v),
            "q_t": torch.from_numpy(q_t),
            "y": torch.tensor([y], dtype=torch.float32),
        }

        nb_v = np.stack([self._id_to_v[int(n)].astype(np.float32) for n in nb_ids], axis=0)
        nb_t = np.stack([self._id_to_t[int(n)].astype(np.float32) for n in nb_ids], axis=0)
        nb_y = np.asarray(
            [self._id_to_y[int(n)] for n in nb_ids],
            dtype=np.float32,
        ).reshape(self.k, 1)
        out["nb_v"] = torch.from_numpy(nb_v)
        out["nb_t"] = torch.from_numpy(nb_t)
        out["w"] = torch.from_numpy(w.astype(np.float32))
        out["nb_y"] = torch.from_numpy(nb_y)
        out["q_pre"] = torch.tensor([float(self._id_to_pre[qid])], dtype=torch.float32)
        return out
