# -*- coding: utf-8 -*-
import os

new_q3 = r"""
\subsection{分析比较}

\subsubsection{Pareto 最优性分析}

为量化``时间与其他指标''之间的权衡，对 2026-08-03 单日以加权单目标形式扫描 7 组 $(w_T, w_{U})$ 权重组合（保留为对比实验，不作为主结果）。如表~\ref{tab:pareto} 所示，时间优先与极端时间优先两个方案位于\textbf{Pareto 前沿}\cite{pareto1906}（共 2 个前沿解）；偏离时间权重会以总时间上升为代价换取利用率改善，反之亦然。更重要的是，主结果两层框架在 08-03 得到 $T^* = 4791$ min，\textbf{严格优于所有 7 个加权方案的最短时间（5099 min）}，验证了``先严格保时间、再在无上界差下优化二级指标''的必要性。

\begin{longtable}{cccccc}
\caption{Q3 单日 Pareto 前沿扫描结果} \label{tab:pareto} \\
\toprule
$w_T$ & $w_{U}$ & 时间(min) & 利用率 & 架次 & Pareto \\
\midrule
\endfirsthead
\toprule
$w_T$ & $w_{U}$ & 时间(min) & 利用率 & 架次 & Pareto \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
\textbf{1.0} & \textbf{0.05} & \textbf{5272} & \textbf{57.03\%} & \textbf{29} & \textbf{是} \\
1.0 & 0.10 & 6557 & 51.27\% & 35 & 否 \\
1.0 & 0.15 & 7058 & 49.94\% & 36 & 否 \\
1.0 & 0.25 & 6963 & 51.98\% & 35 & 否 \\
1.0 & 0.40 & 7074 & 49.97\% & 36 & 否 \\
0.5 & 0.30 & 7432 & 49.60\% & 36 & 否 \\
\textbf{2.0} & \textbf{0.05} & \textbf{5099} & \textbf{55.39\%} & \textbf{28} & \textbf{是} \\
\end{longtable}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\textwidth]{../figures/Problem3/q3_pareto_frontier.png}
\caption{Pareto 前沿散点图（7 组权重）}
\label{fig:q3_pareto}
\end{figure}

\subsubsection{拉格朗日下界认证}

松弛机队约束（式~\eqref{eq:q3_fleet}），引入拉格朗日乘子 $\lambda_{a,t,\tau} \geq 0$\cite{geoffrion1974}，原问题下界为：
\begin{equation}
\text{LB}(\boldsymbol{\lambda}) = \min_{\mathbf{x}, \mathbf{y}} \left\{ Z(\mathbf{x}, \mathbf{y}) + \sum_{a,t,\tau} \lambda_{a,t,\tau} \!\left(\sum_{\substack{r \in \mathcal{R}^{a,t}_d \\ \tau \in [t^{dep}_r,\, t^{ret}_r+30]}} x_r - |\mathcal{M}_{a,t}|\right) \right\}
\label{eq:lagrangian}
\end{equation}

对固定 $\boldsymbol{\lambda}$，子问题保持整数性（非 LP 松弛），用次梯度法\cite{held1974}更新乘子：$\lambda^{k+1}_{a,t,\tau} = \max(0,\, \lambda^k_{a,t,\tau} + \alpha_k g^k_{a,t,\tau})$，步长采用 Polyak 步长 $\alpha_k = \beta_k \cdot (\text{UB} - \text{LB}^{*}) / \|\mathbf{g}\|^2$。

\begin{longtable}{lcc}
\caption{三种下界对比（2026-08-03 单日）} \label{tab:lagrangian} \\
\toprule
方法 & 下界值 & 最优性间隙 \\
\midrule
\endfirsthead
\toprule
方法 & 下界值 & 最优性间隙 \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
MIP 当前解 & 8065.95 & — \\
COPT 内置下界 & 1588.94 & 80.30\% \\
拉格朗日下界 & $-590.51$ & 107.32\% \\
LP 松弛下界 & $-6167.28$ & 176.46\% \\
\end{longtable}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\textwidth]{../figures/Problem3/q3_lagrangian_convergence.png}
\caption{拉格朗日次梯度法收敛曲线}
\label{fig:q3_lagrangian}
\end{figure}

由表~\ref{tab:lagrangian} 可得出三层递进的认证结论。其一，拉格朗日下界（$-590.51$）显著优于 LP 松弛下界（$-6167.28$），证明保持子问题整数性的松弛策略有效；其二，COPT 内置下界最紧（间隙 80.30\%），因其使用完整 MIP 求解器的割平面和分支策略；其三，\textbf{核心瓶颈在 MIP 求解器间隙（60--90\%），而非建模方法}——这解释了下文进阶方法改善有限的原因。
"""

with open(r'D:\数模\26策联杯\8.13MM_STJU\body\q3_part4.tex', 'w', encoding='utf-8') as f:
    f.write(new_q3)
print('Part 4 saved')
