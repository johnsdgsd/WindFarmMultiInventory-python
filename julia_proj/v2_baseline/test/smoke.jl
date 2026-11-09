# smoke.jl — 冒烟：T=2、场景池前 4 条，建图 + SDDP.train 1 轮。
# 只调用 SDDP.jl 内置训练，不实现自定义引擎。
include(joinpath(@__DIR__, "..", "run.jl"))
main(T = 2, pool_size = 4, iterations = 1)
