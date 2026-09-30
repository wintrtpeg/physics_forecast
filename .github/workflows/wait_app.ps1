# run_pforecast.bat 이 띄운 앱(127.0.0.1:8765)이 응답할 때까지 기다린다.
# 실행 창이 먼저 끝나면(설치 실패 등) 로그를 보이고 실패로 끝낸다.
$deadline = (Get-Date).AddMinutes(25)
$launcher = [int](Get-Content launcher.pid)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/api/workspace' -TimeoutSec 5
        if ($r.StatusCode -eq 200) { "앱이 떴습니다: $($r.Content.Substring(0, [Math]::Min(200, $r.Content.Length)))"; exit 0 }
    } catch { }
    if (-not (Get-Process -Id $launcher -ErrorAction SilentlyContinue)) {
        "실행 창이 먼저 끝났습니다 (설치·실행 실패)"
        "---- launcher.out.log"; Get-Content -Encoding utf8 launcher.out.log -ErrorAction SilentlyContinue
        "---- launcher.err.log"; Get-Content -Encoding utf8 launcher.err.log -ErrorAction SilentlyContinue
        exit 1
    }
    Start-Sleep -Seconds 5
}
"25분 안에 뜨지 않았습니다"
"---- launcher.out.log"; Get-Content -Encoding utf8 launcher.out.log -ErrorAction SilentlyContinue
exit 1
