# run_pforecast.bat 을 띄우고, 뜰 때까지 기다린 뒤, API(·화면) 점검과 두 번째 실행까지 **한 단계 안에서** 한다.
# (단계가 끝나면 그 단계의 콘솔이 닫히며 백그라운드로 띄운 앱도 같이 끝난다 — 단계를 나누면 안 된다.)
param([switch]$Ui)

$env:PF_NO_BROWSER = '1'
$env:PF_NO_PAUSE = '1'

function Show-Logs([string]$prefix) {
    Write-Host "---- $prefix.out.log"
    Get-Content -Encoding utf8 "$prefix.out.log" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
    Write-Host "---- $prefix.err.log"
    Get-Content -Encoding utf8 "$prefix.err.log" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
}

function Start-Launcher([string]$prefix, [int]$port) {
    $env:PF_PORT = "$port"
    $p = Start-Process -FilePath cmd.exe -ArgumentList '/c', 'run_pforecast.bat' `
        -RedirectStandardOutput "$prefix.out.log" -RedirectStandardError "$prefix.err.log" -PassThru -WindowStyle Hidden
    $null = $p.Handle          # 핸들을 잡아 둬야 끝난 뒤 ExitCode 를 읽을 수 있다
    return $p
}

function Wait-App($p, [int]$port, [double]$minutes, [string]$prefix) {
    $deadline = (Get-Date).AddMinutes($minutes)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/workspace" -TimeoutSec 5
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        if ($p.HasExited) {
            Write-Host "실행 창이 먼저 끝났습니다 (종료 코드 $($p.ExitCode))"
            Show-Logs $prefix
            return $false
        }
        Start-Sleep -Seconds 5
    }
    Write-Host "$minutes 분 안에 뜨지 않았습니다"
    Show-Logs $prefix
    return $false
}

$fail = 0
$t0 = Get-Date
$p = Start-Launcher 'launcher' 8765
if (-not (Wait-App $p 8765 25 'launcher')) { exit 1 }
Write-Host "앱이 떴습니다: $([int]((Get-Date) - $t0).TotalSeconds)초 (첫 설치 포함)"
Show-Logs 'launcher'

Write-Host "`n==== API 점검"
& .venv\Scripts\python.exe scripts\windows_smoke.py --root .
if ($LASTEXITCODE -ne 0) { $fail = 1 }

if ($Ui) {
    Write-Host "`n==== 화면 점검 (Edge)"
    & .venv\Scripts\python.exe -m pip install --quiet playwright
    & .venv\Scripts\python.exe scripts\windows_ui_smoke.py --channel msedge --shots ui_shots
    if ($LASTEXITCODE -ne 0) { $fail = 1 }
}

Write-Host "`n==== 두 번째 실행 (설치를 건너뛰고 바로 떠야 한다)"
$t1 = Get-Date
$p2 = Start-Launcher 'second' 8766
if (Wait-App $p2 8766 2 'second') {
    Write-Host "두 번째 실행: $([int]((Get-Date) - $t1).TotalSeconds)초 만에 떴습니다"
    Show-Logs 'second'
} else { $fail = 1 }

exit $fail
