"""Command execution on the target, one persistent session per agent.

SSH mode multiplexes over a per-agent ControlMaster socket (OpenSSH built-in),
so each command reuses one TCP+auth session with no paramiko dependency and no
passwords (key-based only). Local mode runs on this box for self-tests.

The executor is content-neutral: it runs whatever command string it is given
and returns the raw result. Compression to facts happens in compress.py.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Protocol


class Target(Protocol):
    """The subset of the harness's TargetConfig the executor needs. Declared
    structurally so this layer imports nothing from swarm/ — any object with
    these attributes (the harness's TargetConfig has them) works."""
    mode: str
    port: int
    key: str | None
    @property
    def ssh_dest(self) -> str: ...


@dataclass
class ExecResult:
    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Executor:
    def start(self) -> None: ...
    def run(self, cmd: str, timeout: float = 30.0, shell: bool = True) -> ExecResult: ...
    def close(self) -> None: ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


def _s(v) -> str:
    """Coerce captured output to str. On TimeoutExpired, subprocess may hand back
    bytes even with text=True, so always normalize."""
    if v is None:
        return ""
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


class LocalExecutor(Executor):
    """Runs commands on this machine. Used both for harness self-tests (shell on,
    for pipes/redirs) and as the sandbox's backend (shell OFF, so metacharacters
    can't slip past the allowlist — see toolmcp.sandbox)."""

    def run(self, cmd: str, timeout: float = 30.0, shell: bool = True) -> ExecResult:
        argv = ["/bin/sh", "-c", cmd] if shell else shlex.split(cmd)
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return ExecResult(cmd, p.returncode, _s(p.stdout), _s(p.stderr))
        except subprocess.TimeoutExpired as e:
            return ExecResult(cmd, 124, _s(e.stdout), _s(e.stderr), timed_out=True)
        except FileNotFoundError:
            return ExecResult(cmd, 127, "", f"{cmd.split()[0] if cmd.split() else cmd}: not found")


class SSHExecutor(Executor):
    """Persistent SSH session via OpenSSH ControlMaster multiplexing."""

    def __init__(self, target: "Target", agent_id: str):
        self.t = target
        self.agent_id = agent_id
        self._sock = os.path.join(
            tempfile.gettempdir(), f"swarm-{os.getuid()}-{agent_id}.sock")
        self._base = [
            "ssh",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self._sock}",
            "-o", "ControlPersist=300",
            "-o", "BatchMode=yes",            # never prompt; fail if a key isn't available
            "-o", "ConnectTimeout=10",        # fail fast on an unreachable target, don't hang the swarm
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(self.t.port),
        ]
        if self.t.key:
            self._base += ["-i", os.path.expanduser(self.t.key)]

    def start(self) -> None:
        # Open the master connection up front so the agent holds one session.
        subprocess.run(
            self._base + ["-o", "ControlPersist=300", "-fN", self.t.ssh_dest],
            check=True, timeout=30)

    def run(self, cmd: str, timeout: float = 30.0, shell: bool = True) -> ExecResult:
        # shell is accepted for interface parity; the remote login shell always
        # interprets the command line, so there is no no-shell mode over SSH.
        argv = self._base + [self.t.ssh_dest, "--", cmd]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return ExecResult(cmd, p.returncode, _s(p.stdout), _s(p.stderr))
        except subprocess.TimeoutExpired as e:
            return ExecResult(cmd, 124, _s(e.stdout), _s(e.stderr), timed_out=True)

    def close(self) -> None:
        subprocess.run(self._base + ["-O", "exit", self.t.ssh_dest],
                       capture_output=True, timeout=10)


def make_executor(target: "Target", agent_id: str) -> Executor:
    # "external" probes a remote host over the network from THIS box (ping/curl/nc),
    # so it executes locally too — no SSH, no key. "local" is the self-test path.
    if target.mode in ("local", "external"):
        return LocalExecutor()
    return SSHExecutor(target, agent_id)
