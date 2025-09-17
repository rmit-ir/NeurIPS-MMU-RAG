#!/usr/bin/env python3
"""
AWS EC2 Proxy Server

This script manages an AWS EC2 instance and runs a Caddy reverse proxy to forward
traffic from a local port to the remote instance.

Usage:
    python proxy.py --remote-instance-id i-04fab8448e7b48317 --remote-instance-region us-west-2 --port 8080 --remote-ip 100.67.56.71 --remote-port 9091
"""

import argparse
from pathlib import Path
import signal
import subprocess
import sys

from tools.aws_tools import (
    describe_instance,
    start_instance,
    stop_instance,
    wait_for_instance_state,
    get_instance_ips
)


class ProxyServer:
    def __init__(self, instance_id, region, host, port, remote_ip, remote_port, remote_ip_type):
        self.instance_id = instance_id
        self.region = region
        self.host = host
        self.port = port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.remote_ip_type = remote_ip_type
        self.caddy_process = None

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)

    def ensure_instance_running(self):
        """Ensure the EC2 instance is running, start it if necessary"""
        print(
            f"Checking status of instance {self.instance_id} in region {self.region}...")

        instance_data = describe_instance(self.instance_id, self.region)
        current_state = instance_data['State']['Name']

        print(f"Instance current state: {current_state}")

        if current_state == 'running':
            print("Instance is already running")
            return
        elif current_state == 'stopped':
            print("Instance is stopped, starting it...")
            start_instance(self.instance_id, self.region)
            print("Waiting for instance to be running...")
            wait_for_instance_state(self.instance_id, self.region, ['running'])
            print("Instance is now running")
        elif current_state == 'terminated':
            raise RuntimeError(
                f"Instance {self.instance_id} is terminated and cannot be started")
        else:
            # Instance is in transitional state (starting, stopping, etc.)
            print(
                f"Instance is in '{current_state}' state, waiting for actionable state...")
            wait_for_instance_state(self.instance_id, self.region, [
                                    'running', 'stopped', 'terminated'])
            # Recursively check again
            self.ensure_instance_running()

    def get_remote_ip(self):
        """Get the IP address of the instance based on remote_ip_type if remote_ip is not provided"""
        if self.remote_ip:
            return self.remote_ip

        print(f"Getting {self.remote_ip_type} IP of the instance...")
        instance_ips = get_instance_ips(self.instance_id, self.region)

        if self.remote_ip_type == "public":
            ip_address = instance_ips.public
            if not ip_address:
                raise RuntimeError(
                    f"Instance {self.instance_id} does not have a public IP address")
        else:  # private
            ip_address = instance_ips.private
            if not ip_address:
                raise RuntimeError(
                    f"Instance {self.instance_id} does not have a private IP address")

        print(f"Instance {self.remote_ip_type} IP: {ip_address}")
        return ip_address

    def start_caddy_proxy(self):
        """Start the Caddy reverse proxy"""
        remote_ip = self.get_remote_ip()

        # Create config directory
        config_dir = Path(
            f"/tmp/caddy_forward_port_{remote_ip}_{self.remote_port}")
        config_dir.mkdir(parents=True, exist_ok=True)

        # Create Caddyfile
        caddyfile_path = config_dir / "Caddyfile"
        caddyfile_content = f"""# Caddy reverse proxy configuration
# Forward traffic from {self.host}:{self.port} to {remote_ip}:{self.remote_port}
{{
    auto_https off
    admin off
}}
:{self.port} {{
    reverse_proxy {remote_ip}:{self.remote_port}
}}
"""
        # ignoring self.host to avoid issues with binding to https

        with open(caddyfile_path, 'w') as f:
            f.write(caddyfile_content)

        # Then run caddy fmt /tmp/testCaddyfile --overwrite
        subprocess.run(["caddy", "fmt", str(caddyfile_path),
                       "--overwrite"], check=True)

        print(f"Created Caddyfile at: {caddyfile_path}")

        # Use Caddy with config file
        caddy_cmd = [
            "caddy",
            "run",
            "--config",
            str(caddyfile_path),
        ]

        print(f"Starting Caddy reverse proxy: {' '.join(caddy_cmd)}")
        print(
            f"Forwarding http://{self.host}:{self.port} -> http://{remote_ip}:{self.remote_port}")

        try:
            self.caddy_process = subprocess.Popen(caddy_cmd)
            print(f"Caddy proxy started with PID {self.caddy_process.pid}")
            return self.caddy_process
        except FileNotFoundError:
            raise RuntimeError(
                "Caddy is not installed or not in PATH. Please install Caddy first.")

    def shutdown(self):
        """Shutdown the proxy and stop the remote instance"""
        print("Shutting down proxy server...")

        # Stop Caddy process
        if self.caddy_process and self.caddy_process.poll() is None:
            print("Stopping Caddy process...")
            self.caddy_process.terminate()
            try:
                self.caddy_process.wait(timeout=10)
                print("Caddy process stopped")
            except subprocess.TimeoutExpired:
                print("Caddy process did not stop gracefully, killing it...")
                self.caddy_process.kill()

        # Stop the remote instance
        print(f"Stopping remote instance {self.instance_id}...")
        try:
            stop_instance(self.instance_id, self.region)
            print("Remote instance stop command sent")
        except Exception as e:
            print(f"Error stopping remote instance: {e}")

    def run(self):
        """Main run loop"""
        try:
            # Ensure instance is running
            self.ensure_instance_running()

            # Start Caddy proxy
            caddy_process = self.start_caddy_proxy()

            print("Proxy server is running. Press Ctrl+C to stop.")

            # Wait for Caddy process to finish or be interrupted
            try:
                caddy_process.wait()
            except KeyboardInterrupt:
                print("\nReceived interrupt signal")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            self.shutdown()


def main():
    parser = argparse.ArgumentParser(description="AWS EC2 Proxy Server")
    parser.add_argument(
        "--remote-instance-id",
        required=True,
        help="AWS EC2 instance ID to manage"
    )
    parser.add_argument(
        "--remote-instance-region",
        required=True,
        help="AWS region where the instance is located"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Local host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Local port to listen on"
    )
    parser.add_argument(
        "--remote-ip",
        help="Remote IP address (if not provided, will use instance's IP based on --remote-ip-type)"
    )
    parser.add_argument(
        "--remote-ip-type",
        choices=["public", "private"],
        default="public",
        help="Type of IP address to use when --remote-ip is not provided (default: public)"
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        required=True,
        help="Remote port to forward traffic to"
    )

    args = parser.parse_args()

    proxy = ProxyServer(
        instance_id=args.remote_instance_id,
        region=args.remote_instance_region,
        host=args.host,
        port=args.port,
        remote_ip=args.remote_ip,
        remote_port=args.remote_port,
        remote_ip_type=args.remote_ip_type,
    )

    proxy.run()


if __name__ == "__main__":
    main()
