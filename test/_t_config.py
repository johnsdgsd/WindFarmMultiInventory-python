# 配置生效性：改 TOML 就一定改到行为；缺键必须报错；派生值按规则换算。
#   python test/_t_config.py
#
# 这条是针对历史 bug 的回归：以前默认值写在 run.py 的 main() 签名里，命令行位置参数
# 会盖掉它，出现过"改了 120 却不生效"。现在唯一事实源是 config/*.toml。

import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from code.config import (ProblemConfig, SolverConfig, load_problem_config,  # noqa: E402
                         load_solver_config, physical_cores)

ROWS = []


def check(name, got, want, note=""):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-12
    ROWS.append((name, "PASS" if ok else "FAIL", got, want, note))
    return ok


def check_raises(name, fn, exc=Exception, note=""):
    try:
        fn()
    except exc:
        ROWS.append((name, "PASS", "已报错", exc.__name__, note))
        return True
    except Exception as e:                                  # 报错类型不对
        ROWS.append((name, "FAIL", type(e).__name__, exc.__name__, note))
        return False
    ROWS.append((name, "FAIL", "未报错", exc.__name__, note))
    return False


tmp = Path(tempfile.mkdtemp(prefix="cfgtest_"))
SRC_P = ROOT / "config/smoke/problem.toml"
SRC_S = ROOT / "config/smoke/solver.toml"

print("=" * 78)
print("§1 读回来的值就是 TOML 里写的值")
print("=" * 78)
p = load_problem_config(SRC_P)
s = load_solver_config(SRC_S)
print(f"  problem: T={p.T} G={p.G} N_R={p.N_R} L={p.L} C={p.C} "
      f"lambda0={p.lambda0} s={p.dirichlet_s} pool={p.pool_size}")
print(f"  solver : iterations={s.iterations} 进程={s.n_processes} "
      f"倍率={s.forward_paths_per_process} 场景={s.n_backward_samples}")
check("problem.T", p.T, 8)
check("problem.G", p.G, 4)
check("problem.N_R", p.N_R, 3)
check("problem.lambda0", p.lambda0, 0.2)
check("problem.dirichlet_s", p.dirichlet_s, 20.0)
check("problem.init_mix 长度 == G", len(p.init_mix), p.G)
check("problem.N 长度 == N_R", len(p.N), p.N_R)
check("problem.fleet_tol", p.fleet_tol, 1e-6)
check("problem.state_round_digits", p.state_round_digits, 6)
check("problem.state_repair_tol", p.state_repair_tol, 1e-4)
check("solver.iterations", s.iterations, 10)
check("solver.n_processes", s.n_processes, 2)

print()
print("=" * 78)
print("§2 改 TOML 的值 ⇒ 读回来的值随之改变（默认值没有第二处）")
print("=" * 78)
for key, old, new in [("T", 8, 9), ("lambda0", 0.2, 0.5),
                      ("C", 50.0, 77.0), ("dirichlet_s", 20.0, 25.0)]:
    txt = SRC_P.read_text(encoding="utf-8").replace(f"{key} = {old}", f"{key} = {new}", 1)
    f = tmp / f"p_{key}.toml"
    f.write_text(txt, encoding="utf-8")
    q = load_problem_config(f)
    got = getattr(q, key)
    print(f"  {key}: {old} → {new}  读到 {got}")
    check(f"改 {key} 生效", got, new)

# fleet_tol 的 TOML 字面量是 1e-6，f"{1e-6}" 会渲染成 1e-06 —— 单独用字面量替换
txt = SRC_P.read_text(encoding="utf-8").replace("fleet_tol = 1e-6",
                                                "fleet_tol = 2e-6", 1)
f = tmp / "p_fleet_tol.toml"
f.write_text(txt, encoding="utf-8")
q = load_problem_config(f)
print(f"  fleet_tol: 1e-6 → 2e-6  读到 {q.fleet_tol}")
check("改 fleet_tol 生效", q.fleet_tol, 2e-6)

txt = SRC_P.read_text(encoding="utf-8").replace("state_round_digits = 6",
                                                "state_round_digits = 7", 1)
f = tmp / "p_state_round_digits.toml"
f.write_text(txt, encoding="utf-8")
q = load_problem_config(f)
check("改 state_round_digits 生效", q.state_round_digits, 7)

txt = SRC_P.read_text(encoding="utf-8").replace("state_repair_tol = 1e-4",
                                                "state_repair_tol = 2e-4", 1)
f = tmp / "p_state_repair_tol.toml"
f.write_text(txt, encoding="utf-8")
q = load_problem_config(f)
check("改 state_repair_tol 生效", q.state_repair_tol, 2e-4)

# 注意：TOML 里写的是 1e-6，而 f"{1e-6}" 会渲染成 1e-06 —— 替换串必须用字面量
for key, old_s, new_s, new in [("iterations", "10", "33", 33),
                               ("n_processes", "2", "5", 5),
                               ("n_simulations", "50", "1234", 1234),
                               ("feas_tol", "1e-6", "1e-8", 1e-8)]:
    txt = SRC_S.read_text(encoding="utf-8").replace(f"{key} = {old_s}",
                                                    f"{key} = {new_s}", 1)
    f = tmp / f"s_{key}.toml"
    f.write_text(txt, encoding="utf-8")
    q = load_solver_config(f)
    got = getattr(q, key)
    print(f"  {key}: {old_s} → {new_s}  读到 {got}")
    check(f"改 {key} 生效", got, new)

# N_R 改了但列表没同步 → 必须报错（而不是静默截断）
txt = SRC_P.read_text(encoding="utf-8").replace("N_R = 3", "N_R = 4", 1)
f = tmp / "bad_nr.toml"
f.write_text(txt, encoding="utf-8")
check_raises("N_R 改了但 N/Qmax/Imax 没同步 → 报错",
             lambda: load_problem_config(f), ValueError)

print()
print("=" * 78)
print("§3 缺键即报错（不静默兜底）")
print("=" * 78)
txt = SRC_P.read_text(encoding="utf-8").replace("N_R = 3\n", "", 1)
f = tmp / "missing.toml"
f.write_text(txt, encoding="utf-8")
check_raises("problem 少 N_R → KeyError", lambda: load_problem_config(f), KeyError)

txt = SRC_P.read_text(encoding="utf-8").replace("fleet_tol = 1e-6\n", "", 1)
f = tmp / "missing_fleet_tol.toml"
f.write_text(txt, encoding="utf-8")
check_raises("problem 少 fleet_tol → KeyError",
             lambda: load_problem_config(f), KeyError)

txt = SRC_P.read_text(encoding="utf-8").replace("state_round_digits = 6\n", "", 1)
f = tmp / "missing_state_round_digits.toml"
f.write_text(txt, encoding="utf-8")
check_raises("problem 少 state_round_digits → KeyError",
             lambda: load_problem_config(f), KeyError)

txt = SRC_P.read_text(encoding="utf-8").replace("state_repair_tol = 1e-4\n", "", 1)
f = tmp / "missing_state_repair_tol.toml"
f.write_text(txt, encoding="utf-8")
check_raises("problem 少 state_repair_tol → KeyError",
             lambda: load_problem_config(f), KeyError)

# 改键名而不是删整行：行尾有注释，按 "n_processes = 2\n" 找是找不到的
txt = SRC_S.read_text(encoding="utf-8").replace("n_processes", "n_procs_renamed", 1)
f = tmp / "missing_s.toml"
f.write_text(txt, encoding="utf-8")
check_raises("solver 少 n_processes → KeyError", lambda: load_solver_config(f), KeyError)

print()
print("=" * 78)
print("§4 派生值换算：n_processes=0 → 物理核//2；路径数 = 倍率 × 进程数")
print("=" * 78)
n_phys = physical_cores()
print(f"  本机物理核 = {n_phys}")
txt = SRC_S.read_text(encoding="utf-8").replace("n_processes = 2", "n_processes = 0", 1)
f = tmp / "auto.toml"
f.write_text(txt, encoding="utf-8")
auto = load_solver_config(f)
print(f"  n_processes = 0 → {auto.n_processes}（= 物理核//2 = {max(1, n_phys // 2)}）")
check("n_processes=0 解析为物理核//2", auto.n_processes, max(1, n_phys // 2))
check("n_forward_paths = 倍率 × 进程数", auto.n_forward_paths,
      auto.forward_paths_per_process * auto.n_processes)

txt = SRC_S.read_text(encoding="utf-8").replace(
    "forward_paths_per_process = 1", "forward_paths_per_process = 3", 1)
f = tmp / "mult.toml"
f.write_text(txt, encoding="utf-8")
m3 = load_solver_config(f)
print(f"  倍率 3 × 进程 {m3.n_processes} → 前向路径 {m3.n_forward_paths}")
check("倍率 3 生效", m3.n_forward_paths, 3 * m3.n_processes)

print()
print("=" * 78)
print("§5 非法值被拦住（带具体数值的报错）")
print("=" * 78)
for i, (label, src, old, new, exc) in enumerate([
        ("G < 2", "problem", "G = 4", "G = 1", ValueError),
        ("lambda0 <= 0", "problem", "lambda0 = 0.2", "lambda0 = 0.0", ValueError),
        ("dirichlet_s <= 0", "problem", "dirichlet_s = 20.0",
         "dirichlet_s = 0.0", ValueError),
        ("p_digits 越界", "problem", "p_digits = 3", "p_digits = 9", ValueError),
        ("init_mix 长度错", "problem",
         "init_mix = [0.5, 0.3, 0.15, 0.05]", "init_mix = [1.0, 0.0, 0.0]", ValueError),
        ("L < 2", "problem", "L = 2", "L = 1", ValueError),
        ("fleet_tol < 0", "problem", "fleet_tol = 1e-6",
         "fleet_tol = -1e-6", ValueError),
        ("fleet_tol > 1e-3", "problem", "fleet_tol = 1e-6",
         "fleet_tol = 1.0", ValueError),
        ("state_round_digits > 12", "problem", "state_round_digits = 6",
         "state_round_digits = 13", ValueError),
        ("state_repair_tol < fleet_tol", "problem", "state_repair_tol = 1e-4",
         "state_repair_tol = 1e-7", ValueError),
        ("state_repair_tol > 1e-2", "problem", "state_repair_tol = 1e-4",
         "state_repair_tol = 1.0", ValueError),
        ("pool_size < 2", "problem", "pool_size = 30", "pool_size = 1", ValueError),
        ("iterations < 0", "solver", "iterations = 10", "iterations = -1", ValueError),
        ("threads < 1", "solver", "threads_per_process = 1",
         "threads_per_process = 0", ValueError),
]):
    base = SRC_P if src == "problem" else SRC_S
    txt = base.read_text(encoding="utf-8").replace(old, new, 1)
    f = tmp / f"bad_{i}.toml"          # 标签里有 < 等字符，不能进文件名
    f.write_text(txt, encoding="utf-8")
    loader = load_problem_config if src == "problem" else load_solver_config
    check_raises(label, lambda f=f, loader=loader: loader(f), exc)

print()
print("=" * 78)
print("§6 dataclasses.replace 会重跑校验（测试里改 T 用的就是它）")
print("=" * 78)
p9 = dataclasses.replace(p, T=9)
check("replace 后 T", p9.T, 9)
check_raises("replace 成非法值也报错",
             lambda: dataclasses.replace(p, lambda0=0.0), ValueError)
check_raises("replace 成非法 fleet_tol 也报错",
             lambda: dataclasses.replace(p, fleet_tol=1.0), ValueError)
check_raises("replace 成非法 state_repair_tol 也报错",
             lambda: dataclasses.replace(p, state_repair_tol=1.0), ValueError)

shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 78)
print("汇总")
print("=" * 78)
n_fail = sum(1 for r in ROWS if r[1] == "FAIL")
for name, st, got, want, note in ROWS:
    line = f"  [{st}] {name}"
    if st == "FAIL":
        line += f"   got={got} 期望={want} {note}"
    print(line)
print(f"  {len(ROWS) - n_fail}/{len(ROWS)} 通过")
sys.exit(1 if n_fail else 0)
