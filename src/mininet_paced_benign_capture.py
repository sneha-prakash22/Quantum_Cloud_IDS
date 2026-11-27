import subprocess
import time
from pathlib import Path

from mininet.net import Mininet
from mininet.topo import SingleSwitchTopo
from mininet.node import OVSBridge

PROJECT = Path("/mnt/c/Users/rojav/OneDrive/Documents/Quantum_Cloud_IDS")
OUTPUT = PROJECT / "data" / "mininet_validation"
PCAP = OUTPUT / "mininet_benign_tcp_paced_001.pcap"
LOG = OUTPUT / "mininet_benign_traffic_paced_001.log"

PORT = 5002
TOTAL_BYTES = 500_000
CHUNK_BYTES = 1000
DELAY_SECONDS = 0.02

SERVER_CODE = """
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", int(sys.argv[1])))
s.listen(1)
c, address = s.accept()
total = 0
while True:
    data = c.recv(65536)
    if not data:
        break
    total += len(data)
print(f"RECEIVED_BYTES={total}", flush=True)
c.close()
s.close()
"""

CLIENT_CODE = """
import socket, sys, time
host = sys.argv[1]
port = int(sys.argv[2])
total = int(sys.argv[3])
chunk_size = int(sys.argv[4])
delay = float(sys.argv[5])
s = socket.create_connection((host, port))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
chunk = b"X" * chunk_size
sent = 0
start = time.time()
while sent < total:
    amount = min(chunk_size, total - sent)
    s.sendall(chunk[:amount])
    sent += amount
    time.sleep(delay)
s.shutdown(socket.SHUT_WR)
elapsed = time.time() - start
s.close()
print(f"SENT_BYTES={sent}")
print(f"GENERATION_SECONDS={elapsed:.3f}")
print(f"APPLICATION_RATE_BYTES_PER_SECOND={sent / elapsed:.2f}")
"""

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for path in (PCAP, LOG):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing experiment file: {path}"
            )

    net = None
    capture = None
    server = None
    client = None
    capture_error = ""
    server_output = ""
    client_output = ""
    log_lines = []

    try:
        net = Mininet(
            topo=SingleSwitchTopo(2),
            switch=OVSBridge,
            controller=None
        )
        net.start()

        h1 = net.get("h1")
        h2 = net.get("h2")
        h1_ip = h1.IP()
        h2_ip = h2.IP()

        log_lines.extend([
            "Experiment: Paced controlled benign TCP traffic",
            "Topology: Two hosts connected through OVSBridge",
            f"Source host: h1 ({h1_ip})",
            f"Destination host: h2 ({h2_ip})",
            "Capture interface: h1-eth0",
            f"TCP destination port: {PORT}",
            "Traffic generator: Python TCP socket",
            f"Target application bytes: {TOTAL_BYTES}",
            f"Chunk size: {CHUNK_BYTES}",
            f"Delay between chunks: {DELAY_SECONDS} seconds",
        ])

        print("Checking host connectivity...")
        packet_loss = net.pingAll()
        log_lines.append(
            f"Pre-capture connectivity packet loss: {packet_loss}%"
        )
        if packet_loss != 0:
            raise RuntimeError(
                f"Connectivity check failed: {packet_loss}% packet loss"
            )

        print("Disabling TSO and GSO on both hosts...")
        for host in (h1, h2):
            result = host.cmd(
                "ethtool -K h1-eth0 tso off gso off"
                if host.name == "h1"
                else "ethtool -K h2-eth0 tso off gso off"
            )
            features = host.cmd("ethtool -k " + host.name + "-eth0")
            if "tcp-segmentation-offload: off" not in features:
                raise RuntimeError(
                    f"TSO could not be verified off on {host.name}"
                )
            if "generic-segmentation-offload: off" not in features:
                raise RuntimeError(
                    f"GSO could not be verified off on {host.name}"
                )
            log_lines.append(
                f"{host.name} ethtool command output: {result.strip() or 'No error reported'}"
            )
            log_lines.append(f"{host.name} TSO/GSO verified off")

        print("Starting packet capture...")
        capture = h1.popen(
            [
                "tcpdump", "-U", "-B", "4096",
                "-i", "h1-eth0", "-nn", "-s", "0",
                "-w", str(PCAP),
                f"tcp port {PORT}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(2)
        if capture.poll() is not None:
            _, capture_error = capture.communicate()
            raise RuntimeError(
                f"tcpdump exited before traffic generation: {capture_error}"
            )

        print("Starting controlled TCP receiver...")
        server = h2.popen(
            ["python3", "-u", "-c", SERVER_CODE, str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        time.sleep(1)
        if server.poll() is not None:
            server_output, _ = server.communicate()
            raise RuntimeError(
                f"TCP receiver exited unexpectedly: {server_output}"
            )

        print("Generating 500 KB of paced TCP traffic...")
        client = h1.popen(
            [
                "python3", "-u", "-c", CLIENT_CODE,
                h2_ip, str(PORT), str(TOTAL_BYTES),
                str(CHUNK_BYTES), str(DELAY_SECONDS)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        try:
            client_output, _ = client.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            client.terminate()
            try:
                client_output, _ = client.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                client.kill()
                client_output, _ = client.communicate()
            raise RuntimeError("TCP sender exceeded the 40-second timeout")

        log_lines.extend([
            "",
            "=== TCP SENDER OUTPUT ===",
            client_output,
            f"TCP sender exit code: {client.returncode}"
        ])

        if client.returncode != 0:
            raise RuntimeError(
                f"TCP sender failed with exit code {client.returncode}"
            )

        try:
            server_output, _ = server.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            raise RuntimeError("TCP receiver did not finish within 10 seconds")

        log_lines.extend([
            "",
            "=== TCP RECEIVER OUTPUT ===",
            server_output
        ])

        expected = f"RECEIVED_BYTES={TOTAL_BYTES}"
        if expected not in server_output:
            raise RuntimeError(
                f"Receiver byte count did not match expected value: {expected}"
            )

        print("Traffic generation and byte-count verification succeeded.")
        time.sleep(2)

    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            try:
                client_output, _ = client.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                client.kill()
                client_output, _ = client.communicate()

        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server_output, _ = server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server_output, _ = server.communicate()

        if capture is not None and capture.poll() is None:
            capture.terminate()
            try:
                _, capture_error = capture.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                capture.kill()
                _, capture_error = capture.communicate()
        elif capture is not None:
            _, capture_error = capture.communicate()

        if net is not None:
            net.stop()

        log_lines.extend([
            "",
            "=== TCPDUMP CAPTURE SUMMARY ===",
            capture_error or "No tcpdump summary available."
        ])
        LOG.write_text("\n".join(log_lines), encoding="utf-8")

    if not PCAP.exists() or PCAP.stat().st_size <= 24:
        raise RuntimeError(
            "PCAP is missing or contains no data beyond its basic header."
        )

    print("Experiment completed.")
    print(f"PCAP: {PCAP}")
    print(f"PCAP size: {PCAP.stat().st_size} bytes")
    print(f"Traffic log: {LOG}")

if __name__ == "__main__":
    main()
