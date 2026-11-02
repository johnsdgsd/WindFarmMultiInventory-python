# ITER‑SDD\(\mathscr P\) 完整版数学模型、原理与算法
> 包含：原问题、动态规划、自适应分区、**绝对分裂规则、贪心聚类(Algorithm 1原文完整数学描述)**、两层循环完整算法、收敛命题、超参、实现注意事项。
> 前置假设
> 1. **假设1 固定补偿**：补偿矩阵$A_t$、成本向量$c_t$与随机实现无关；不确定性仅来自$B_t(\xi_t),\,b_t(\xi_t)$。
> 2. **假设2 阶段独立**：$\xi_t$独立于历史$\xi_{[t-1]}$。
> 3. **假设3 非退化**：代价‑to‑go函数处处取有限实数值。
> 4. **假设4 相对完全补偿**：任意可行状态下子问题一定可行。

## 1 原问题：多阶段随机线性规划 MSL\(\mathscr P\)
时域 $\(t=1\),2,\dots,T$。
- $\(t=1\)$：确定；决策 $x_1\in\mathbb R_+^{n_1}$；
- $t\ge2$：随机 $\xi_t=(B_t,b_t)\in\Xi_t$；场景$k\in\Xi_t$，概率 $p_t^k\ge0,\ \sum_{k\in\Xi_t}p_t^k=1$；
- $x_t = x_t(\xi_{[t]})$：$t$阶段决策，依赖随机历史$\xi_{[t]}=(\xi_1,\dots,\xi_t)$。

嵌套形式原问题
$$
\begin{aligned}
\min_{
\begin{array}{c}
A_1 x_1 = b_1\\
x_1 \in \mathbb R_+^{n_1}
\end{array}
}
c_1^\top x_1
+
\mathbb E_{\xi_2}\Bigg[
\min_{
\begin{array}{c}
B_2 x_1 + A_2 x_2 = b_2 \\
x_2\in\mathbb R_+^{n_2}
\end{array}
}
c_2^\top x_2
+
\mathbb E_{\xi_3}\Big[\cdots
+
\mathbb E_{\xi_{T-1}}
\Bigg[
\min_{
\begin{array}{c}
B_T x_{T-1}+A_T x_T = b_T\\
x_T\in\mathbb R_+^{n_T}
\end{array}
}
c_T^\top x_T
\Bigg]
\cdots
\Bigg]
\end{aligned}
$$

### 动态规划形式
阶段子问题价值函数，给定状态$x_{t-1}$与随机实现$\xi_t$
$$
Q_t(x_{t-1},\xi_t)
=
\min_{
\begin{array}{c}
x_t\in\mathbb R_+^{n_t} \\
A_t x_t = b_t - B_t x_{t-1}
\end{array}
}
c_t^\top x_t + \mathfrak Q_{t+1}(x_t)
$$
终端条件：$\boldsymbol{\mathfrak Q_{T+1}(\cdot)\equiv 0}$。

期望代价‑to‑go函数
$$
\mathfrak Q_t(x_{t-1})
=
\mathbb E_{\xi_t}\big[ Q_t(x_{t-1},\xi_t) \big]
=\sum_{k\in\Xi_t} p_t^k\, Q_t(x_{t-1},\xi_t^k)
$$

$\mathfrak Q_t(\cdot)$凸分段线性。SDD\(\mathscr P\)使用割平面做下方逼近：
$$
\check{\mathfrak Q}_t(x_{t-1})
=
\max_{j\in J_t}
\Big\{\alpha_{t,j}+\beta_{t,j}^\top x_{t-1}\Big\}
$$
$J_t$：阶段$t$割平面集合；满足 $\check{\mathfrak Q}_t(\cdot)\le \mathfrak Q_t(\cdot)$。

第一阶段主问题，给出原问题下界
$$
\underline z =
\min_{
\begin{array}{c}
A_1 x_1 = b_1 \\
x_1\in\mathbb R_+^{n_1}
\end{array}
}
c_1^\top x_1 + \check{\mathfrak Q}_2(x_1)
$$

## 2 自适应场景分区数学模型
### 2.1 场景分区定义
**定义（阶段$t$场景分区）**
$$
\mathscr N_t = \big\{\mathscr \(\mathscr P\)_t^1,\mathscr \(\mathscr P\)_t^2,\dots,\mathscr \(\mathscr P\)_t^{L_t}\big\}
$$
满足
1. $\bigcup_{\ell=1}^{L_t}\mathscr \(\mathscr P\)_t^\ell = \Xi_t$；
2. $\mathscr \(\mathscr P\)_t^{\ell_1}\cap\mathscr \(\mathscr P\)_t^{\ell_2}=\emptyset,\;\forall \ell_1\neq\ell_2$。

$\mathscr \(\mathscr P\)_t^\ell$：场景簇；$L_t$：簇数量。簇总概率权重：
$$
\bar p_t^\ell = \sum_{k\in \mathscr \(\mathscr P\)_t^\ell} p_t^k
$$

### 2.2 簇聚合代表
$$
\bar B_t^\ell
=
\sum_{k\in \mathscr \(\mathscr P\)_t^\ell}
\frac{p_t^k}{\bar p_t^\ell}\, B_t^k,\quad
\bar b_t^\ell
=
\sum_{k\in \mathscr \(\mathscr P\)_t^\ell}
\frac{p_t^k}{\bar p_t^\ell}\, b_t^k
$$
聚合随机实现 $\bar\xi_t^\ell = (\bar B_t^\ell,\,\bar b_t^\ell)$。

### 2.3 分区子问题与近似代价‑to‑go
簇子问题（使用聚合代表）
$$
\overline Q_t(x_{t-1},\bar\xi_t^\ell)
=
\min_{
\begin{array}{c}
x_t\in\mathbb R_+^{n_t}\\
A_t x_t = \bar b_t^\ell - \bar B_t^\ell x_{t-1}
\end{array}
}
c_t^\top x_t + \check{\mathfrak Q}_{t+1}(x_t)
$$

分区近似期望代价‑to‑go
$$
\overline{\mathfrak Q}_t(x_{t-1})
=
\sum_{\ell=1}^{L_t} \bar p_t^\ell \cdot \overline Q_t(x_{t-1},\bar\xi_t^\ell)
$$

### 2.4 Jensen下界性质（固定补偿假设）
$Q_t(x_{t-1},\xi_t)$对$\xi_t=(B_t,b_t)$凸。
$$
\mathbb E\big[Q_t(x_{t-1},\xi_t)\,\big|\, \xi_t\in\mathscr \(\mathscr P\)_t^\ell\big]
\ge
Q_t\big(x_{t-1},\; \mathbb E[\xi_t\mid \xi_t\in\mathscr \(\mathscr P\)_t^\ell]\big)
=\overline Q_t(x_{t-1},\bar\xi_t^\ell)
$$
求和得到核心不等式
$$
\boldsymbol{\mathfrak Q_t(x_{t-1}) \ge \overline{\mathfrak Q}_t(x_{t-1})}
$$
> 分区近似是真实代价函数的下界；簇生成的粗割是合法Benders最优割。

### 2.5 对偶形式与充分分区引理
固定补偿下，全部场景共享同一个对偶可行域：
$$
\\(\mathscr P\)i_t = \big\{\pi_t \in\mathbb R^{m_t}\,\big|\, A_t^\top\pi_t \le c_t\big\}
$$

场景子问题对偶：
$$
Q_t(x_{t-1},\xi_t^k)
=\max_{\pi_t\in\\(\mathscr P\)i_t} \big(b_t^k-B_t^k x_{t-1}\big)^\top \pi_t
$$

簇代表子问题对偶：
$$
\overline Q_t(x_{t-1},\bar\xi_t^\ell)
=\max_{\pi_t\in\\(\mathscr P\)i_t} \big(\bar b_t^\ell-\bar B_t^\ell x_{t-1}\big)^\top \pi_t
$$

记 $\\(\mathscr P\)i^{*}(x_{t-1},\xi_t^k)$ 为场景$k$的对偶最优解集；
$\bar\pi_t^\ell\in \\(\mathscr P\)i^{*}(x_{t-1},\bar\xi_t^\ell)$为簇代表的一个最优对偶。

> **引理1（Lemma 1，充分分区判据）**
若
$$
\bar\pi_t^\ell \in \bigcap_{k\in \mathscr \(\mathscr P\)_t^\ell} \\(\mathscr P\)i^{*}(x_{t-1},\xi_t^k)
$$
即簇代表的最优对偶同时是簇内**所有原始场景的最优对偶**，则
$$
\mathfrak Q_t(x_{t-1}) = \overline{\mathfrak Q}_t(x_{t-1})
$$
簇不需要分裂；否则 $\mathfrak Q_t(x_{t-1})>\overline{\mathfrak Q}_t(x_{t-1})$，存在间隙，簇需要细化。

### 2.6 粗割平面构造
trial点$\check x_{t-1}$，簇子问题最优对偶$\bar\pi_t^\ell$：
$$
\begin{cases}
\bar\alpha_t^\ell = \big(\bar b_t^\ell - \bar B_t^\ell \check x_{t-1}\big)^\top \bar\pi_t^\ell\\
\bar\beta_t^\ell = -\big(\bar B_t^\ell\big)^\top \bar\pi_t^\ell
\end{cases}
$$

单簇割：$q_t^\ell(x_{t-1})=\bar\alpha_t^\ell+\big(\bar\beta_t^\ell\big)^\top\big(x_{t-1}-\check x_{t-1}\big)$。

概率加权得到全局粗割：
$$
\bar q_t(x_{t-1})
=
\sum_{\ell=1}^{L_t} \bar p_t^\ell \cdot q_t^\ell(x_{t-1})
$$

更新割集合：
$$
\check{\mathfrak Q}_t(\cdot) \leftarrow \max\big\{\check{\mathfrak Q}_t(\cdot),\; \bar q_t(\cdot)\big\}
$$

---

# 3 绝对规则(Absolute Rule) + 贪心聚类（原文Algorithm 1完整版）
> 输入：待分裂旧簇$\mathscr \(\mathscr P\)$；簇内每个场景$k\in\mathscr \(\mathscr P\)$的最优对偶向量$\{\pi^k\}_{k\in\mathscr \(\mathscr P\)}$，子问题目标值$\{Q^k:=Q(x_{t-1},\xi^k)\}_{k\in\mathscr \(\mathscr P\)}$；超参$\epsilon^\pi>0$（对偶欧氏距离阈值），$\epsilon^Q>0$（目标值间隙阈值，处理L\(\mathscr P\)退化多重对偶）。
> 输出：分裂之后的一簇集合 $\mathscr \(\mathscr P\)' = \{\mathscr \(\mathscr P\)'^1,\mathscr \(\mathscr P\)'^2,\dots\}$。

### 分裂判定条件（绝对规则 Absolute Rule）
两个场景$k,k'\in\mathscr \(\mathscr P\)$应当分到不同簇，当且仅当：
$$
\boldsymbol{\big\|\pi^{k}-\pi^{k'}\big\|_2>\epsilon^\pi \quad \textbf{and} \quad \big|Q^{k}-Q^{k'}\big|>\epsilon^Q}
$$
- 如果仅仅对偶差距大，但目标函数值接近，判定为L\(\mathscr P\)对偶退化，**不分裂**，保留同一簇。

### 贪心聚类数学过程（Algorithm 1）
1. 取旧簇内第一个元素 $k_0 \in \mathscr \(\mathscr P\)$；初始化新簇列表 $\mathscr \(\mathscr P\)' \leftarrow \big\{\{k_0\}\big\}$；对应代表对偶集合 $\\(\mathscr P\)i' \leftarrow \{\pi^{k_0}\}$。
2. 遍历旧簇中剩余每一个场景 $k\in \mathscr \(\mathscr P\),\ k\neq k_0$：
    1. 设置标记 $\text{flag}=0$；
    2. 遍历当前已经生成的每一个新簇索引 $m$，其代表对偶为 $\hat\pi^{(m)}\in\\(\mathscr P\)i'$，簇成员集合$\mathscr \(\mathscr P\)'^m$：
        - 检验：$\|\pi^{k}-\hat\pi^{(m)}\|_2 \le \epsilon^\pi$ **or** $|Q^{k}-Q(\hat\pi^{(m)})|\le \epsilon^Q$。
        - 若条件成立：将$k$加入$\mathscr \(\mathscr P\)'^m$；$\text{flag}\leftarrow 1$；跳出内层循环。
    3. 若 $\text{flag}=0$：
        - 新建簇 $\mathscr \(\mathscr P\)'^{\text{new}}=\{k\}$；
        - $\mathscr \(\mathscr P\)' \leftarrow \mathscr \(\mathscr P\)'\cup\{\mathscr \(\mathscr P\)'^{\text{new}}\}$；
        - $\\(\mathscr P\)i' \leftarrow \\(\mathscr P\)i'\cup\{\pi^{k}\}$。
3. 全部场景遍历完成，返回细化后的簇集合$\mathscr \(\mathscr P\)'$。

> 复杂度：对大小为$|\mathscr \(\mathscr P\)|$的簇，复杂度$\mathcal O(|\mathscr \(\mathscr P\)|^2)$；整个分区细化总复杂度$\mathcal O\left(\sum_{\mathscr \(\mathscr P\)\in\mathscr N_t}|\mathscr \(\mathscr P\)|^2\right)$。

> 说明：该是启发式贪心算法，不求解N\(\mathscr P\)‑hard最大团问题；牺牲最优聚类换取可接受计算开销。

---

# 4 ITER‑SDD\(\mathscr P\) 完整两层循环算法
符号汇总
- $p$：分区细化迭代序号；$\mathscr N_t^{(p)}$：第$p$次细化后阶段$t$分区；
- $j$：内层SDD\(\mathscr P\)迭代序号；$\underline z^{(j)}$：第$j$轮SDD\(\mathscr P\)下界；
- $n$：连续观测迭代数；$\epsilon_{stab}>0$：下界停滞阈值；
- Quick‑\(\mathscr P\)ass(Q\(\mathscr P\))树遍历：前向 $1\to 2\to\cdots\to T$；后向 $T\to T-1\to\cdots\to1$；阶段之间不做来回迭代。

## Step 0 初始化
1. 对 $t=2,\dots,T$：初始分区，全部场景合并为单个簇
$$
\mathscr N_t^{(0)}=\big\{\,\Xi_t\,\big\}
$$
2. 初始化割集合 $J_t=\emptyset$，$\check{\mathfrak Q}_t^{(0)}(\cdot)\equiv -\infty$；
3. 设置超参：$\epsilon^\pi,\epsilon^Q,\,n,\,\epsilon_{stab}$；
4. $p\leftarrow 0$。

## Step 1 内层循环：在当前粗分区$\{\mathscr N_t^{(p)}\}$运行 Quick‑\(\mathscr P\)ass SDD\(\mathscr P\)
循环执行SDD\(\mathscr P\)迭代 $j=1,2,3,\dots$：
1. **前向\(\mathscr P\)ass**：在粗场景树上采样1条样本路径；依次求解各阶段问题，得到trial状态序列 $\check x_1,\check x_2,\dots,\check x_{T-1}$。
2. **后向\(\mathscr P\)ass**，$\(t=T\)$ downto $2$：
    - 遍历本阶段每一个簇 $\mathscr \(\mathscr P\)_t^\ell\in \mathscr N_t^{(p)}$，求解**分区簇子问题**，获取簇最优对偶$\bar\pi_t^\ell$；
    - 按公式构造粗割$\bar q_t(\cdot)$；将粗割加入割集合$J_t$，更新$\check{\mathfrak Q}_t(\cdot)$。
3. 求解第一阶段主问题，记录本轮下界 $\underline z^{(j)}$。
4. **停滞判定条件**：
$$
\underline z^{(j)} - \underline z^{(j-n)} \;\le\; \epsilon_{stab}
$$
    - 条件**不成立**：继续内层SDD\(\mathscr P\)迭代；
    - 条件**成立**：退出内层循环，进入独立分区细化 Step 2。

## Step 2 独立分区细化 Refinement Step（不嵌入SDD\(\mathscr P\) backward pass）
1. 从**原始全场景树**采样样本路径 $\xi^s=(\xi_2^{s_2},\dots,\xi_T^{s_T})$，获得trial点序列 $\{\check x_{t-1}\}_{t=2,\dots,T}$。
2. 对每个阶段 $t=2,\dots,T$：
    1. 取出当前分区 $\mathscr N_t^{(p)}$。
    2. 对每一个旧簇 $\mathscr \(\mathscr P\)\in \mathscr N_t^{(p)}$：
        - 求解簇内**全部原始场景子问题**，得到每个$k\in\mathscr \(\mathscr P\)$：最优对偶$\pi_t^k$，价值$Q_t(\check x_{t-1},\xi_t^k)$；
        - 调用【绝对规则+贪心聚类】子过程，输入$\mathscr \(\mathscr P\),\{\pi_t^k\},\{Q_t^k\},\epsilon^\pi,\epsilon^Q$，输出分裂后的新簇集合$\mathscr \(\mathscr P\)'$。
    3. 收集全部分裂输出的新簇，组成细化后的阶段$t$分区 $\mathscr N_t^{(p+1)}$。
3. 更新细化计数器 $p\leftarrow p+1$，得到整套新分区$\{\mathscr N_t^{(p+1)}\}$。

## Step 3 全局终止判断
- 若满足全局停止准则（上下界统计间隙、最大时间、最大迭代次数），算法终止，输出决策策略；
- 否则，回到 **Step 1**，在新粗分区$\{\mathscr N_t^{(p)}\}$继续内层SDD\(\mathscr P\)。

> ✅关键架构特点：**细化步骤独立于SDD\(\mathscr P\)的backward pass**。只有当下界提升停滞，才触发代价高昂的全场景子问题求解，避免频繁分裂造成簇数量爆炸。

## 5 ITER‑SDD\(\mathscr P\)收敛命题（\(\mathscr P\)roposition 1）
**假设**
1. 全部4条基础假设成立；
2. SDD\(\mathscr P\)采用**有放回采样**；
3. 细化规则：只要 $\check{\mathfrak Q}_t(\check x_{t-1}) \neq \mathbb E[\underline Q_t(\check x_{t-1},\xi_t)]$，簇的数量严格增大；直到 $L_t=|\Xi_t|$（每个簇退化为单一场景）。

**结论**：依概率1，有限次前向‑后向迭代，算法得到原MSL\(\mathscr P\)的最优策略。

> 证明梗概：
> 每个阶段簇数目存在上界$|\Xi_t|$，簇数量不能无限增长。迭代最终两种结局：
> 1. 得到$\epsilon$‑充分分区，分区近似等价原问题；
> 2. 分区退化为全部单场景，ITER‑SDD\(\mathscr P\)等价标准SDD\(\mathscr P\)，继承标准SDD\(\mathscr P\)收敛性质。

## 6 超参说明表
|参数|物理含义|
|---|---|
|$\epsilon^\pi>0$|对偶向量欧氏距离，簇分裂距离阈值|
|$\epsilon^Q>0$|子问题目标值差阈值；处理L\(\mathscr P\)多重最优对偶（退化）|
|$n\in \mathbb N^+$|下界停滞判断，连续观测SDD\(\mathscr P\)迭代轮数|
|$\epsilon_{stab}>0$|下界停滞阈值，控制何时触发分区细化|

## 7 工程实现关键约束与坑点
1. **固定补偿是硬前提**：$A_t$或$c_t$若随场景改变，Jensen不等式不再成立，粗割不再保证下界，整套算法理论失效。
2. 细化Step 2计算代价高：需要求解簇内部**全部原始L\(\mathscr P\)子问题**；ITER‑SDD\(\mathscr P\)设计目标就是尽量少进入细化步骤。
3. 初始化策略：全部场景合并为1个簇，最大化早期迭代加速效果。
4. 树遍历：只用Quick‑\(\mathscr P\)ass(Q\(\mathscr P\))，不要混入Cautious‑\(\mathscr P\)ass(C\(\mathscr P\))，C\(\mathscr P\)会带来巨大计算开销。
5. 退化防护：**不能只依靠对偶距离做分裂，必须同时使用$\epsilon^Q$做目标值校验**，否则L\(\mathscr P\)多重对偶解会造成不必要大量簇分裂。

如果你下一步需要，我可以输出：
1. ITER‑SDD\(\mathscr P\)可直接用于编程的伪代码（类\(\mathscr P\)ython伪代码，含贪心聚类子过程）；
2. 区分：ITER‑SDD\(\mathscr P\) / A\(\mathscr P\)E\(\mathscr P\)‑SDD\(\mathscr P\) / A\(\mathscr P\)‑Q\(\mathscr P\)‑SDD\(\mathscr P\) 的对比简表。



参数	符号	含义	调参影响 & 风险	经验参考值		
对偶距离分裂阈值	ϵπ	两个场景对偶向量π的欧氏距离阈值；绝对规则里，距离超过该值才有可能触发分裂。	过大：簇很少分裂，分区一直很粗，粗割偏差大，下界提升慢，收敛慢。
过小：轻微对偶差异就分裂，簇快速膨胀接近全场景，加速效果消失，退化成标准 SDDP。	10−2∼10−4；根据对偶向量量级缩放；优先看问题对偶的数值尺度，不要直接死套 1e‑4。		
目标函数退化阈值	ϵQ	子问题价值Qt​差值阈值；用于处理 LP 退化（多重最优对偶）。即使对偶距离大，如果价值差小于ϵQ，不分裂簇。	过大：本该分裂的簇不分裂，保留过大簇，近似误差大。
过小：失去退化防护，LP 多重对偶会产生大量不必要簇分裂，簇数量爆炸。	与优化问题目标量级相关，一般取原问题单阶段成本的 10−3∼10−5。		
连续观测迭代轮数	n	判断下界停滞：看最近n轮 SDDP 下界变化。	n太大：要跑很多轮才触发细化，长期带着误差很大的粗割跑，收敛慢。
n太小：频繁触发细化，细化步骤开销很大，总耗时暴涨。	n=5∼15，论文实验隐含用的n=10。		
下界停滞阈值	ϵstab​	触发分区细化的条件：z​(j)−z​(j−n)≤ϵstab​，连续n轮下界提升不超过该值，则执行细化。	ϵstab​过大：很容易触发细化，细化步骤频繁执行，大量时间花在求解全部原始场景子问题。
ϵstab​过小：很难触发细化，长期使用粗糙分区，近似间隙一直存在，下界停滞不动。	按原问题最优值的相对值设置，推荐相对阈值 $\epsilon_{stab}= 10^{-2}\cdot	z^*_{\text{est}}	$；不知道最优值，取目标量级的 $10^{-2}\sim10^{-3}$。