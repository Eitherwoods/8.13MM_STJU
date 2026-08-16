# -*- coding: utf-8 -*-
import os

new_q3 = r"""
\subsection{模型建立}

\subsubsection{问题分解与预处理}

问题三的规模是问题二的 7 倍（7 天 $\times$ 4000 人 $\times$ 24 架飞机），直接求解不可行。但从题目约束中可提炼出关键洞察，实现严格等价的降维分解。

\begin{theorem}
\label{thm:decouple}
给定约束``飞机不得在海上设施过夜、每个架次必须返回所属机场、飞机 06:00--18:00 起飞且不晚于 20:00 返回''，任意架次的起飞时刻与返回时刻必在同一自然日内。
\end{theorem}

\begin{proof}
设架次在 $d$ 日 $t$ 时刻起飞（$t \geq 06{:}00$）。若返回时刻落在 $d+1$ 日，则飞机在海上设施停留跨过午夜，违反``不得在海上设施过夜''；若飞机在返回前停留于机场，则架次已在 $d$ 日结束。故架次不可能跨天。
\end{proof}

\begin{corollary}
\label{cor:indep}
由于架次不跨天、飞机每日从所属机场重置出发，各日之间不存在状态转移耦合。多日排班问题可精确分解为 $D = 7$ 个独立的单日问题：
$\min \sum_{d=1}^{7} Z_d$，其中 $Z_d$ 为第 $d$ 日最优目标值。
\end{corollary}

\begin{corollary}
\label{cor:dateset}
对需求 $p$，其可行日期集合为 $D_p = \{ d : [e_p, l_p] \cap [06{:}00, 20{:}00]_d \neq \emptyset \}$。由于 84.1\% 的需求时间窗跨天，$|D_p| \geq 2$ 常成立，为需求-日期分配提供自由度。
\end{corollary}

在此基础上，进一步给出时间窗的预处理方法。

\begin{proposition}
\label{prop:cpm}
对候选路径 $r = [A, S_1, \ldots, S_K, A]$，定义前向最早到达时刻与后向最晚离开时刻：
\begin{equation}
\text{EAT}_k = \max\!\left(\text{EAT}_{k-1} + \frac{d_{k-1,k}}{v} \times 60 + s_{k-1},\; \max_{p \in P_k^{\mathrm{b}}} e_p\right)
\label{eq:eat}
\end{equation}
\begin{equation}
\text{LDT}_k = \min\!\left(\text{LDT}_{k+1} - \frac{d_{k,k+1}}{v} \times 60,\; \min_{p \in P_k^{\mathrm{a}}} l_p\right)
\label{eq:ldt}
\end{equation}

其中 $s_k$ 为停靠时间（加油 20 min / 普通 10 min），$v$ 为机型速度，边界条件 $\text{EAT}_0 = 06{:}00$，$\text{LDT}_{K+1} = 20{:}00$。若 $\exists\, k$ 使得 $\text{EAT}_k > \text{LDT}_k$，则路径 $r$ 不可行，直接剔除。此收缩为纯数学预处理，以 $O(K)$ 代价在路径生成阶段剔除时间窗不可行路径。
\end{proposition}

此外，按天解耦的贪心分配跨天需求存在可量化的损失上界：设跨天需求占比为 $\alpha$，单日 MIP 相对间隙为 $\delta$，单条跨天需求分配到非最优日期的额外成本上界为 $\Delta$，则 $\hat{T} \leq (1+\delta) T^* + \alpha |\mathcal{P}| \Delta$。本数据 $\alpha = 84.1\%$，MIP 间隙是主要瓶颈而非解耦损失。

\subsubsection{时间扩展网络与统一建模}

以 10 min 粒度将 $[06{:}00, 20{:}00]$ 离散化为 84 个时间槽\cite{ford1962}。每个机场 $\times$ 机型 $\times$ 时间槽对应一个资源节点，架次占用其起飞至返回后 30 min 周转期内的连续槽位，将机队约束、运营时段、周转时间统一编码为网络结构。

复用问题一/二的 Clarke-Wright 节省算法生成空间路径，叠加 14 个均匀分布的起飞时刻偏移（0--660 min，间隔约 60 min），对每个需求检查时间窗兼容性，仅对兼容的 $(r, p)$ 对创建变量。路径池规模控制在约 9400 条。

\subsubsection{个人级覆盖约束的修正}

在问题三的初步求解结果中，通过独立校验发现名义方案虽然满足 MIP 报告的全部约束（违反量为零），但存在 \textbf{39 条需求未被任何选中架次覆盖}的现象。进一步统计表明，这 39 条未覆盖需求全部属于应急处置（emergency）与增储上产（production）任务，即时间窗最紧、业务优先级最高的两类需求。

原模型的需求覆盖约束为设施级人数约束：
\begin{equation}
\sum_{r \in \mathcal{R}} y^O_{r,f} = n^O_f \quad \forall f
\end{equation}

该约束只保证``运往设施 $f$ 的总人数等于需求数''，而将同一设施的所有需求视为同质。当设施 $f$ 的需求时间窗差异较大时，求解器可以选择仅覆盖``时间窗宽松需求''的晚起飞路径，而漏掉``时间窗紧需求''。

本文采用三层修正，在不改变设施级变量结构的前提下补全个人级覆盖保证：

\textbf{(1) 个人级覆盖约束。} 对每条需求 $p$，引入约束
\begin{equation}
\sum_{r \in \mathcal{R}_p} x_r \geq 1 \quad \forall p
\end{equation}
其中 $\mathcal{R}_p$ 为与需求 $p$ 时间窗兼容的候选路径集合。

\textbf{(2) 紧需求优先分配。} 在人员分配阶段，对同一设施的候选需求按最晚到达时间升序排序。

\textbf{(3) 紧急穿梭直达路径。} 对应急处置与增储上产的穿梭需求，强制生成``去程直达 + 返程段加油''的专用路径。

修正效果如表~\ref{tab:q3_correction} 所示。

\begin{table}[htbp]
\centering
\caption{个人级覆盖约束修正效果}
\label{tab:q3_correction}
\begin{tabular}{lcc}
\toprule
指标 & 修正前 & 修正后 \\
\midrule
未覆盖需求数 & 39 & 0 \\
MIP 最优性间隙 & 109\% & 61\% \\
座位利用率 & 43.2\% & 49.0\% \\
加权目标值 & 8340 & 5879 \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(r'D:\数模\26策联杯\8.13MM_STJU\body\q3_part2.tex', 'w', encoding='utf-8') as f:
    f.write(new_q3)
print('Part 2 saved')
