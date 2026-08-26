import os
import re
import time
import socket
import signal
import subprocess
from pathlib import Path

COMFY_PORT = 8188
TEST_PORT = 8765
CF = "/content/cloudflared"


def sh(cmd, timeout=30):
    print(f"\n$ {cmd}")
    try:
        p = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
        if p.stdout:
            print(p.stdout[-12000:])
        if p.stderr:
            print("[stderr]")
            print(p.stderr[-12000:])
        print("exit:", p.returncode)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        print("ERROR:", repr(e))
        return -1, "", repr(e)


def kill_matching(pattern):
    subprocess.run(f"pkill -f '{pattern}'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)


def parse_url(log_path):
    txt = Path(log_path).read_text(errors="ignore") if Path(log_path).exists() else ""
    urls = re.findall(r"https://[-a-z0-9]+\.trycloudflare\.com", txt)
    return (urls[-1] if urls else None), txt


def start_quick_tunnel(origin, tag, protocol=None):
    log_path = f"/content/cloudflared_{tag}.log"
    pid_path = f"/content/cloudflared_{tag}.pid"
    try:
        Path(log_path).unlink()
    except FileNotFoundError:
        pass

    cmd = [CF, "tunnel", "--url", origin, "--no-autoupdate", "--loglevel", "debug"]
    if protocol:
        cmd += ["--protocol", protocol]

    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    Path(pid_path).write_text(str(p.pid))

    url = None
    for _ in range(40):
        time.sleep(0.5)
        if p.poll() is not None:
            break
        url, _ = parse_url(log_path)
        if url:
            break

    print(f"\n[{tag}] pid={p.pid} process_alive={p.poll() is None} origin={origin}")
    print(f"[{tag}] url={url}")
    if url:
        host = url.split("//", 1)[1]
        sh(f"getent ahostsv4 {host} | head")
        sh(f"curl -sv --connect-timeout 10 --max-time 20 {url}/ -o /content/{tag}_body.html -D /content/{tag}_headers.txt", timeout=25)
        print(f"[{tag}] headers:")
        if Path(f"/content/{tag}_headers.txt").exists():
            print(Path(f"/content/{tag}_headers.txt").read_text(errors="ignore")[-5000:])
        print(f"[{tag}] body head:")
        if Path(f"/content/{tag}_body.html").exists():
            print(Path(f"/content/{tag}_body.html").read_text(errors="ignore")[:1500])

    print(f"[{tag}] cloudflared log tail:")
    _, txt = parse_url(log_path)
    print(txt[-12000:])
    return p, url, log_path


print("=" * 80)
print("A. Environment")
print("=" * 80)
sh("date -u")
sh("uname -a")
sh(f"{CF} --version")
sh("python --version")
sh("ip addr show | sed -n '1,120p'")
sh("cat /etc/resolv.conf")
sh("env | grep -Ei 'https?_proxy|no_proxy|cloudflare' || true")

print("=" * 80)
print("B. ComfyUI local origin")
print("=" * 80)
sh(f"ss -ltnp | grep ':{COMFY_PORT} ' || true")
sh(f"curl -sv --connect-timeout 5 --max-time 10 http://127.0.0.1:{COMFY_PORT}/ -o /dev/null")
sh(f"curl -sv --connect-timeout 5 --max-time 10 http://localhost:{COMFY_PORT}/ -o /dev/null")
sh(f"curl -sS --max-time 10 http://127.0.0.1:{COMFY_PORT}/object_info | head -c 300")

print("=" * 80)
print("C. Kill old cloudflared processes")
print("=" * 80)
kill_matching("cloudflared tunnel")
sh("pgrep -af cloudflared || true")

print("=" * 80)
print("D. Control experiment: plain Python HTTP server through Quick Tunnel")
print("=" * 80)
http_log = open("/content/test_http_server.log", "w")
http_p = subprocess.Popen(["python", "-m", "http.server", str(TEST_PORT), "--bind", "127.0.0.1", "--directory", "/content"], stdout=http_log, stderr=subprocess.STDOUT)
time.sleep(1)
sh(f"curl -sv --max-time 10 http://127.0.0.1:{TEST_PORT}/ -o /dev/null")
cf_test, test_url, test_log = start_quick_tunnel(f"http://127.0.0.1:{TEST_PORT}", "control", protocol="http2")

print("=" * 80)
print("E. ComfyUI through a fresh Quick Tunnel")
print("=" * 80)
try:
    cf_test.terminate()
    cf_test.wait(timeout=5)
except Exception:
    pass

time.sleep(2)
cf_comfy, comfy_url, comfy_log = start_quick_tunnel(f"http://127.0.0.1:{COMFY_PORT}", "comfy", protocol="http2")

print("=" * 80)
print("F. Interpretation")
print("=" * 80)
print("control_url:", test_url)
print("comfy_url  :", comfy_url)
print("\n판정 기준:")
print("1) control도 530/403 -> ComfyUI와 무관. Colab↔Cloudflare Quick Tunnel 경로/Quick Tunnel 서비스 문제.")
print("2) control=200, ComfyUI만 실패 -> ComfyUI origin/header/websocket/host 처리 문제 쪽.")
print("3) 둘 다 200 -> 기존 cloudflared 프로세스/생성된 hostname이 꼬였던 것.")
print("4) cloudflared log에 'Unable to reach the origin service'가 있으면 localhost origin 연결 문제.")
print("5) 외부 curl 응답 body의 Cloudflare error code와 cf-ray를 같이 보낼 것.")
print("\ncomfy cloudflared PID는 살아있게 두었습니다:", cf_comfy.pid)
print("접속 주소:", comfy_url)
