# io.py — 结果目录、配置、割文件与训练日志的读写。与问题解耦，可整目录迁移。
"""结果目录约定（与 Julia 版一致）：

    results/<tag>_<yyyymmdd_HHMMSS>/
        config.txt             key=value，结果目录自描述
        problem_data.json      问题数据（含预采样场景池）
        cuts.json              Benders 割（Cut 的完整记录）
        sddp_training_log.csv  训练日志（空格分隔，兼容 plot/plot_bounds.py）
        summary_<时间戳>.txt   下界/上界/间隙报表

只新增、不覆盖：训练与仿真只往结果目录里写新文件，或先写 .tmp 再原子改名。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from .cut import FORMAT_VERSION, Cut

LOG_HEADER = (" iteration    simulation      bound        time (s)"
              "     solves  pid")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def new_run_dir(tag: str, root) -> Path:
    """在 root 下建 `results/<tag>_<时间戳>/` 并返回路径。"""
    d = Path(root) / f"{(tag or 'run')}_{timestamp()}"
    d.mkdir(parents=True, exist_ok=False)
    return d


# --------------------------------------------------------------------- 配置
def write_config(d, pairs) -> Path:
    path = Path(d) / "config.txt"
    with open(path, "w", encoding="utf-8") as f:
        for k, v in pairs:
            f.write(f"{k}={v}\n")
    return path


def read_config(d) -> dict:
    path = Path(d) / "config.txt"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ----------------------------------------------------------- 原子写（割文件）
def _atomic_write_text(path: Path, text: str) -> None:
    """先写 .tmp 再改名。割文件可达数百 MB，被中途强杀时不能毁掉上一份完好的。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化类型: {type(o)}")


def save_json(d, name: str, obj) -> Path:
    path = Path(d) / name
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, default=_json_default))
    return path


def load_json(d, name: str):
    path = Path(d) / name
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- 割文件
def save_cuts(d, cuts_by_stage, meta: dict, name: str = "cuts.json") -> Path:
    """`cuts_by_stage`：list，下标 t=0..T，元素为 [Cut, ...]（期 0 为空）。"""
    payload = {
        "meta": dict(meta),
        "version": FORMAT_VERSION,
        "cuts": {str(t): [c.to_dict() for c in cuts_by_stage[t]]
                 for t in range(len(cuts_by_stage)) if cuts_by_stage[t]},
    }
    path = Path(d) / name
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, default=_json_default))
    return path


def load_cuts(d, t_max: int, dim: int, name: str = "cuts.json") -> tuple[list, dict]:
    """读回割，返回 (cuts_by_stage, meta)。"""
    payload = load_json(d, name)
    cuts = [[] for _ in range(t_max + 1)]
    for k, items in payload.get("cuts", {}).items():
        t = int(k)
        if not 0 <= t <= t_max:
            raise ValueError(f"割文件里的期号 {t} 超出 0..{t_max}")
        cuts[t] = [Cut.from_dict(c) for c in items]
    for t in range(t_max + 1):
        for c in cuts[t]:
            if c.dim != dim:
                raise ValueError(f"割文件状态维度 {c.dim} ≠ 当前 {dim}（模型已改，不可 resume）")
    return cuts, payload.get("meta", {})


def count_cuts(cuts_by_stage) -> int:
    return int(sum(len(c) for c in cuts_by_stage))


# ------------------------------------------------------------------- 训练日志
class TrainingLog:
    """空格分隔的训练日志，表头与 SDDP.jl 一致，plot/plot_bounds.py 可直接解析。"""

    def __init__(self, d, name: str = "sddp_training_log.csv"):
        self.path = Path(d) / name
        self._f = open(self.path, "a", encoding="utf-8")
        if self.path.stat().st_size == 0:
            self._f.write("-" * 68 + "\n")
            self._f.write(LOG_HEADER + "\n")
            self._f.write("-" * 68 + "\n")
            self._f.flush()

    def row(self, iteration: int, simulation: float, bound: float,
            elapsed: float, solves: int, pid: int | None = None) -> None:
        self._f.write(f"{iteration:10d}   {simulation:.6e}  {bound:.6e}  "
                      f"{elapsed:.6e}  {solves:10d}  {pid if pid is not None else os.getpid():5d}\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def archive_log(d, prefix: str = "sddp_training_log") -> Path | None:
    """续训前把旧日志改名归档，避免被追写混淆。"""
    src = Path(d) / f"{prefix}.csv"
    if not src.is_file():
        return None
    dst = Path(d) / f"{prefix}_prev_{timestamp()}.csv"
    os.replace(src, dst)
    return dst


def save_summary(d, text: str, name: str | None = None) -> Path:
    path = Path(d) / (name or f"summary_{timestamp()}.txt")
    path.write_text(text, encoding="utf-8")
    return path


def list_result_dirs(root) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
