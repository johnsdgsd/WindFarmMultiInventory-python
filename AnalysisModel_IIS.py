# LP 不可行冲突集分析：读入 .lp，判为不可行则计算 IIS 并写出。
#
# 用法：
#   python AnalysisModel_IIS.py <现场.lp> [--solver copt|gurobi] [-o 输出文件]
#
#   --solver / -s   gurobi | copt，默认 copt
#   -o / --output   输出文件名；不传则 gurobi → gurobi_infeas.ilp、copt → copt_infeas.iis
#                   （Gurobi 按扩展名挑写出格式，文件名要带 .ilp）
#
# 现场 .lp 由 utils/solver.py 的 export_lp 在子问题求解失败时自动导出到结果目录
# （subproblem_<环节>_t<期>_s<场景>.lp，环节 = forward / backward / lower_bound）。
#
# 例：
#   python test/_t_iis.py                                   # 造一个不可行子问题
#   python AnalysisModel_IIS.py test/_iis_probe.lp -s copt  # 分析它的冲突集

import argparse
import os
import gurobipy as gp
import coptpy as cp

def analyze_with_gurobi(lp_file: str, out_iis: str = "gurobi_infeas.ilp") -> str:
    """Gurobi读取lp，求解，不可行则生成IIS文件，返回输出路径"""
    if not os.path.isfile(lp_file):
        raise FileNotFoundError(f"LP文件不存在: {lp_file}")

    model = gp.read(lp_file)
    model.setParam("OutputFlag", 1)
    model.optimize()

    status = model.status
    if status == gp.GRB.INFEASIBLE:
        print(f"[Gurobi] 模型不可行，正在计算IIS...")
        model.computeIIS()
        model.write(out_iis)
        full_out = os.path.abspath(out_iis)
        print(f"[Gurobi] IIS不可行子集已保存至: {full_out}")
        return full_out
    elif status == gp.GRB.OPTIMAL:
        print("[Gurobi] 模型可行，存在最优解，无需生成IIS")
        return ""
    else:
        print(f"[Gurobi] 求解状态码: {status}，非可行/不可行")
        return ""


def analyze_with_copt(lp_file: str, out_iis: str = "copt_infeas.iis") -> str:
    if not os.path.isfile(lp_file):
        raise FileNotFoundError(f"LP文件不存在: {lp_file}")

    env = cp.Envr()
    try:
        model = env.createModel("copt_iis_analyze")
        model.readLp(lp_file)
        model.solve()

        # 直接使用 model.status 获取状态，匹配官方常量
        status = model.status
        if status == cp.COPT.INFEASIBLE:
            print(f"[COPT] 模型不可行，正在计算IIS...")
            model.computeIIS()
            model.writeIIS(out_iis)
            full_out = os.path.abspath(out_iis)
            print(f"[COPT] IIS不可行子集已保存至: {full_out}")
            return full_out
        elif status == cp.COPT.OPTIMAL:
            print("[COPT] 模型可行，存在整数最优解，无需生成IIS")
            return ""
        elif status == cp.COPT.UNBOUNDED:
            print("[COPT] 模型无界，无法生成IIS")
            return ""
        elif status == cp.COPT.UNSTARTED:
            print("[COPT] 模型未执行求解")
            return ""
        else:
            print(f"[COPT] 未知求解状态，status={status}")
            return ""
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="LP模型不可行IIS分析工具（支持Gurobi/COPT）")
    parser.add_argument("lp_path", help="待分析的 .lp 文件完整路径")
    # 默认求解器改为 copt
    parser.add_argument("--solver", "-s", choices=["gurobi", "copt"], default="copt", help="选择求解器，默认copt")
    parser.add_argument("--output", "-o", help="自定义IIS输出文件名，默认自动生成")
    args = parser.parse_args()

    lp_path = args.lp_path
    solver = args.solver
    out_file = args.output

    try:
        if solver == "gurobi":
            if out_file is None:
                out_file = "gurobi_infeas.ilp"
            res_path = analyze_with_gurobi(lp_path, out_file)
        else:
            if out_file is None:
                out_file = "copt_infeas.iis"
            res_path = analyze_with_copt(lp_path, out_file)

        if res_path:
            print(f"\n执行完成，IIS结果文件: {res_path}")
        else:
            print("\n执行完成，模型无不可行问题，未输出IIS文件")

    except Exception as e:
        print(f"\n程序异常: {str(e)}")


if __name__ == "__main__":
    main()