import subprocess
import time
from pathlib import Path

from mininet.net import Mininet
from mininet.topo import SingleSwitchTopo
from mininet.node import OVSBridge

PROJECT = Path("/mnt/c/Users/rojav/OneDrive/Documents/Quantum_Cloud_IDS")
OUTPUT = PROJECT / "data" / "mininet_validation"
PCAP = OUTPUT / "mininet_benign_tcp_ovsbridge.pcap"
LOG = OUTPUT / "mininet_benign_traffic_ovsbridge.log"

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
    log_lines = []
    capture_error = ""
    server_output = ""
    client_output = ""

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
        interface = "h1-eth0"
        port = 5001

        log_lines.extend([
            "Experiment: Controlled benign TCP traffic",
            "Topology: Two hosts connected through OVSBridge",
            f"Source host: h1 ({h1_ip})",
            f"Destination host: h2 ({h2_ip})",
            f"Capture interface: {interface}",
            f"TCP destination port: {port}",
            "Traffic generator: iperf 2",
            "Client duration: 15 seconds",
        ])

        print("Checking host connectivity...")
        packet_loss = net.pingAll()
        log_lines.append(f"Pre-capture connectivity packet loss: {packet_loss}%")

        if packet_loss != 0:
            raise RuntimeError(
                f"Connectivity check failed: {packet_loss}% packet loss"
            )

        print("Starting packet capture...")
        capture = h1.popen(
            [
                "tcpdump", "-U", "-i", interface,
                "-nn", "-s", "0",
                "-w", str(PCAP),
                f"tcp port {port}"
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

        print("Starting iperf server...")
        server = h2.popen(
            ["iperf", "-s", "-p", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        time.sleep(2)

        if server.poll() is not None:
            server_output, _ = server.communicate()
            raise RuntimeError(
                f"iperf server exited before the client connected: {server_output}"
            )

        print("Generating controlled TCP traffic...")
        client = h1.popen(
            [
                "iperf", "-c", h2_ip,
                "-p", str(port),
                "-t", "15",
                "-i", "1"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        try:
            client_output, _ = client.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            client.terminate()
            try:
                client_output, _ = client.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                client.kill()
                client_output, _ = client.communicate()
            raise RuntimeError("iperf client exceeded the 25-second timeout")

        log_lines.extend([
            "",
            "=== IPERF CLIENT OUTPUT ===",
            client_output,
            f"iperf client exit code: {client.returncode}"
        ])

        if client.returncode != 0:
            raise RuntimeError(
                f"iperf client failed with exit code {client.returncode}"
            )

        print("Traffic generation completed successfully.")
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
            "=== IPERF SERVER OUTPUT ===",
            server_output or "No server output available.",
            "",
            "=== TCPDUMP CAPTURE SUMMARY ===",
            capture_error or "No tcpdump summary available."
        ])

        LOG.write_text("\n".join(log_lines), encoding="utf-8")

    if not PCAP.exists() or PCAP.stat().st_size <= 24:
        raise RuntimeError(
            "PCAP is missing or contains no data beyond its basic header. "
            "Inspect the capture log before proceeding."
        )

    print("Experiment completed.")
    print(f"PCAP: {PCAP}")
    print(f"PCAP size: {PCAP.stat().st_size} bytes")
    print(f"Traffic log: {LOG}")

if __name__ == "__main__":
    main()
