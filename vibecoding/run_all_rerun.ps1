# 文件: run_all_rerun.ps1
# 功能: 按计划文档顺序重跑 Q1/Q2/Q3 全部求解器与辅助分析，单步失败不终止
# 用法: powershell -ExecutionPolicy Bypass -File run_all_rerun.ps1

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'

$root = 'D:\数模\26策联杯\8.13MM_STJU'
$py = 'E:\Python\python.exe'
$logs = Join-Path $root 'logs'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# ---------- 0. 备份现有输出 ----------
Write-Output '=== 备份旧输出 ==='
$outDirs = @(
    (Join-Path $root 'output\Problem1'),
    (Join-Path $root 'output\Problem2'),
    (Join-Path $root 'output\Problem3'),
    (Join-Path $root 'output\Problem1_lexicographic'),
    (Join-Path $root 'output\Problem2_lexicographic'),
    (Join-Path $root 'output\Problem1_weighted'),
    (Join-Path $root 'output\Problem2_weighted')
)
foreach ($d in $outDirs) {
    if (-not (Test-Path -LiteralPath $d)) { continue }
    Get-ChildItem -LiteralPath $d -Recurse -File | Where-Object {
        $_.Name -like 'q1-*' -or $_.Name -like 'q2-*' -or $_.Name -like 'q3-*'
    } | ForEach-Object {
        $bak = $_.FullName + ".bak_$ts"
        Copy-Item -LiteralPath $_.FullName -Destination $bak -Force
    }
}
Write-Output '=== 备份完成 ==='

# ---------- 工具函数 ----------
function Run-Step {
    param([string]$Name, [string]$WorkDir, [string]$Script, [string[]]$Args)
    $log = Join-Path $logs "${Name}_${ts}.log"
    $start = Get-Date -Format 'HH:mm:ss'
    Write-Output "=== $Name 开始 $start ==="
    Push-Location $WorkDir
    if ($Args.Count -gt 0) {
        & $py $Script @Args *> $log
    } else {
        & $py $Script *> $log
    }
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date -Format 'HH:mm:ss'
    Write-Output "=== $Name 结束 code=$code $end ==="
    return $code
}

$results = @()

# 4.1 基础三问
$results += Run-Step -Name 'problem1_solution' -WorkDir (Join-Path $root 'vibecoding\problem12') -Script 'problem1_solution.py'
$results += Run-Step -Name 'problem2_solution' -WorkDir (Join-Path $root 'vibecoding\problem12') -Script 'problem2_solution.py'
$results += Run-Step -Name 'problem3_solution' -WorkDir (Join-Path $root 'vibecoding\problem3') -Script 'problem3_solution.py'

# 4.2 加权单目标
$results += Run-Step -Name 'weighted_solver' -WorkDir (Join-Path $root 'vibecoding\problem12') -Script 'weighted_solver.py'

# 4.3 两阶段 epsilon-tolerance
$results += Run-Step -Name 'two_stage_solver' -WorkDir (Join-Path $root 'vibecoding\problem12') -Script 'two_stage_solver.py'

# 4.4 ALNS 初始解改进（消融实验入口）
$results += Run-Step -Name 'validate_alns_0803' -WorkDir (Join-Path $root 'vibecoding\problem3') -Script 'validate_alns_0803.py'

# 4.5 Pareto 前沿
$results += Run-Step -Name 'pareto_sweep' -WorkDir (Join-Path $root 'vibecoding\problem3') -Script 'pareto_sweep.py'

# 4.6 鲁棒性分析
$results += Run-Step -Name 'robust_dynamic_test' -WorkDir (Join-Path $root 'vibecoding\problem3') -Script 'robust_dynamic_test.py'

# 4.7 统一验证（必跑）
$results += Run-Step -Name 'validate_all' -WorkDir (Join-Path $root 'vibecoding') -Script 'validate_all.py' -Args @('--all')

# ---------- 汇总 ----------
Write-Output ''
Write-Output '==================== 重跑汇总 ===================='
for ($i = 0; $i -lt $results.Count; $i++) {
    Write-Output "步骤 $($i+1): code=$($results[$i])"
}
