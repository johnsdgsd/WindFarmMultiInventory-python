# _t_fcmc_resume.py — 固定簇 meta 身份校验 + simulate.py 端到端元数据。
#   python test/_t_fcmc_resume.py
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (MulticutConfig, load_problem_config,  # noqa: E402
                         load_solver_config)
from code.problem_data import DefineProblem                      # noqa: E402
from code.subproblem import make_stage_subproblem                # noqa: E402
from utils.engine import SDDPEngine                              # noqa: E402
from utils.io import load_cuts                                   # noqa: E402


def _write_cfg(td: Path, scfg) -> None:
    """结果目录的配置副本：solver 改成 mode=multi / L=2，供 simulate.py 读回。"""
    dst = td / "config"
    src = ROOT / "config" / "smoke"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "problem.toml", dst / "problem.toml")
    txt = (src / "solver.toml").read_text(encoding="utf-8")
    txt = re.sub(r'^mode\s*=\s*"single"', 'mode = "multi"', txt, count=1,
                 flags=re.M)
    txt = re.sub(r"^n_clusters\s*=\s*\d+", "n_clusters = 2", txt, count=1,
                 flags=re.M)
    txt = re.sub(r'^group_mode\s*=\s*"scene"', 'group_mode = "failure_quantile"',
                 txt, count=1, flags=re.M)
    txt = re.sub(r"^n_processes\s*=\s*\d+", "n_processes = 1", txt, count=1,
                 flags=re.M)
    (dst / "solver.toml").write_text(txt, encoding="utf-8")


def main():
    pcfg = load_problem_config(ROOT / "config" / "smoke" / "problem.toml")
    scfg = load_solver_config(ROOT / "config" / "smoke" / "solver.toml")
    scfg.n_processes = 1
    scfg = dataclasses.replace(
        scfg, mode="multi",
        multicut=MulticutConfig(n_clusters=2, group_mode="failure_quantile"))
    prob = DefineProblem(pcfg)
    T, dim = prob.T, prob.state_dim

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_cfg(td, scfg)
        eng = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=td,
                         log=lambda *a, **k: None)
        try:
            eng.train(1, verbose=False)
        finally:
            eng.close()
        cuts, meta = load_cuts(td, T, dim, expected_mode="multi")
        assert meta["multicut"] == {"n_clusters": 2,
                                    "group_mode": "failure_quantile",
                                    "cluster_sizes": [15, 15],
                                    "cluster_hash": meta["multicut"]["cluster_hash"]}
        assert meta["multicut"]["cluster_hash"]

        for mutate in ("hash", "drop"):
            m = json.loads(json.dumps(meta))
            if mutate == "hash":
                m["multicut"]["cluster_hash"] = "0" * 40
            else:
                m.pop("multicut")
            e = SDDPEngine(prob, make_stage_subproblem, scfg, results_dir=None,
                           log=lambda *a, **k: None)
            try:
                try:
                    e.check_cut_meta(m)
                except SystemExit:
                    pass
                else:
                    raise AssertionError(f"meta 变异 {mutate} 应被拒绝")
            finally:
                e.close()
        print("  [PASS] cluster_hash 不符 / 缺 multicut 块 拒绝加载")

        r = subprocess.run([sys.executable, str(ROOT / "simulate.py"), str(td),
                            "10", "5"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        sim = json.loads((td / "sim_10paths.json").read_text(encoding="utf-8"))
        assert sim["multicut"]["n_clusters"] == 2
        assert sim["mode"] == "multi" and sim["n_theta"] == 2
        assert sim["audit_violations"] == {}, sim["audit_violations"]
        assert sim["terms_vs_stage_cost_max_dev"] < 1e-6
        print("  [PASS] simulate.py：10 路径、审计 0 违约、multicut 元数据齐全")

    print("OK _t_fcmc_resume")


if __name__ == "__main__":
    main()
