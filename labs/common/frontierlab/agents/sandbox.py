"""Run untrusted Python in a child process: temp directory, timeout, no network, scrubbed environment.

    from frontierlab.agents.sandbox import run_python
    res = run_python("print(1 + 1)", files={"data.txt": "hello"}, timeout=5)
    res.returncode, res.stdout, res.timed_out          # 0, '2\\n', False

What it does, in order:

1. Creates a fresh temporary directory, writes ``files`` into it (relative paths only; ``..`` and absolute
   paths are refused), and makes it the child's working directory. The directory is deleted afterwards
   (``keep=True`` keeps it, for debugging).
2. Starts ``python -I`` (isolated mode: no user site-packages, ``PYTHON*`` variables ignored, the script's
   folder not on ``sys.path``) with an environment that contains only what the interpreter needs to start
   (``PATH``, ``SYSTEMROOT`` and ``TEMP`` on Windows, ``HOME`` set to the temp directory) plus ``env``.
3. Runs a **guard** before the user code: it replaces ``socket.socket.connect``, ``connect_ex``,
   ``sendto``, ``socket.create_connection`` and ``getaddrinfo`` with functions that raise, and replaces
   ``subprocess.Popen``, ``os.system`` and the ``os.exec*`` / ``os.spawn*`` / ``os.fork`` family the same
   way, so that the child cannot open a connection or start another program through the standard library.
4. On POSIX, sets resource limits in the child before it starts (address space ``memory_mb``, CPU seconds,
   file size, no core dumps) and puts it in its own session so a timeout kills the whole process group. On
   Windows it uses a new process group and kills the process tree on timeout (``taskkill /T /F``).
5. Waits at most ``timeout`` seconds (wall clock), captures stdout and stderr (each cut to
   ``max_output`` characters) and returns a :class:`SandboxResult`, including the files the child left
   behind (``collect=[...]``).

**Limits: this is not a security boundary.** It stops *accidents* and the cheap exploits a small model or a
course exercise produces (endless loops, huge outputs, writing outside the workspace through relative paths,
accidental network use, ``os.system``). It does **not** stop a program that wants out: ``ctypes`` can call
the C library directly, the guard runs in the same interpreter as the code it guards and can be undone by
it, absolute paths can still reach the rest of your disk with your user's permissions, and on Windows there
is no memory or CPU limit (only the wall-clock timeout). Code written by a capable model, or any code you
did not write, belongs in a container or VM with no network and a read-only host mount (for example Docker
with ``--network none --read-only``, gVisor, or a Firecracker microVM) — REASONABLE INDUSTRY PRACTICE, and
what SWE-bench-style harnesses do (one container image per task).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

GUARD = r'''
import builtins, os, socket, subprocess, sys
def _blocked(*a, **k):
    raise PermissionError("blocked by the course sandbox")
for _name in ("connect", "connect_ex", "sendto", "sendall", "send"):
    setattr(socket.socket, _name, _blocked)
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
subprocess.Popen = _blocked
os.system = _blocked
for _name in ("execv", "execve", "execl", "execle", "execlp", "execlpe", "execvp", "execvpe", "spawnl",
              "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "fork", "forkpty",
              "posix_spawn", "posix_spawnp", "startfile", "popen"):
    if hasattr(os, _name):
        setattr(os, _name, _blocked)
del _name
sys.argv = [__SCRIPT__]
sys.path.insert(0, os.getcwd())
import runpy
runpy.run_path(__SCRIPT__, run_name="__main__")
'''


@dataclass
class SandboxResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float
    files: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _safe_rel(path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or str(path).startswith(("/", "\\")):
        raise ValueError(f"refusing to write outside the sandbox: {path!r}")
    return p


def _limits(memory_mb: int, cpu_seconds: int):            # POSIX only, runs in the child before exec
    def apply():
        import resource
        os.setsid()
        if memory_mb:
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 2**20, memory_mb * 2**20))
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**20, 64 * 2**20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _kill_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()


def run_python(code: str | None = None, *, files: dict[str, str] | None = None, script: str = "main.py",
               stdin: str = "", timeout: float = 10.0, max_output: int = 20_000, memory_mb: int = 1024,
               env: dict[str, str] | None = None, collect: list[str] | None = None, keep: bool = False,
               guard: bool = True) -> SandboxResult:
    """Run ``code`` (written to ``script``) or an existing ``script`` from ``files`` in a fresh temp dir."""
    files = dict(files or {})
    if code is not None:
        files[script] = code
    if script not in files:
        raise ValueError(f"no {script!r} to run")
    root = Path(tempfile.mkdtemp(prefix="flsbx_"))
    try:
        for rel, text in files.items():
            dest = root / _safe_rel(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        child_env = {"HOME": str(root), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        for k in ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "COMSPEC"):
            if k in os.environ:
                child_env[k] = os.environ[k]
        child_env.update(env or {})
        if guard:
            boot = root / "_sandbox_boot.py"
            boot.write_text(GUARD.replace("__SCRIPT__", repr(script)), encoding="utf-8")
            argv = [sys.executable, "-I", str(boot)]
        else:
            argv = [sys.executable, "-I", script]
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["preexec_fn"] = _limits(memory_mb, int(timeout) + 1)
        t0 = time.perf_counter()
        proc = subprocess.Popen(argv, cwd=root, env=child_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", **kw)
        timed_out = False
        try:
            out, err = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            out, err = proc.communicate()
        secs = time.perf_counter() - t0
        got = {}
        for rel in collect or []:
            p = root / _safe_rel(rel)
            if p.is_file():
                got[rel] = p.read_text(encoding="utf-8", errors="replace")[:max_output]
        return SandboxResult(None if timed_out else proc.returncode, (out or "")[:max_output],
                             (err or "")[:max_output], timed_out, secs, got)
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_python", "SandboxResult", "GUARD"]
