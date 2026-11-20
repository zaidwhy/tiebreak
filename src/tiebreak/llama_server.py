"""Score with a GGUF reranker through a llama-server subprocess using rank pooling.

Ollama ships llama-server next to its own binary (lib/ollama/llama-server), so a machine
that runs Ollama already has a llama.cpp build that can rerank; nothing is compiled here.
"""
from __future__ import annotations

import atexit
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from tiebreak.engine import DeadlineExceeded, LimitError, Scored
from tiebreak.gguf import SOFTMAX_ARCHS, check_reranker, read_header

log = logging.getLogger("tiebreak.llama_server")

EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"


class LlamaServerError(RuntimeError):
    pass


def find_llama_server() -> Path:
    """$TIEBREAK_LLAMA_SERVER, then llama-server on PATH, then the copy bundled with Ollama."""
    env = os.environ.get("TIEBREAK_LLAMA_SERVER")
    if env:
        path = Path(env)
        if not path.is_file():
            raise LlamaServerError(f"TIEBREAK_LLAMA_SERVER={env} is not a file")
        return path
    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)
    candidates = []
    ollama = shutil.which("ollama")
    if ollama:
        base = Path(ollama).resolve().parent
        candidates += [base / "lib" / "ollama" / EXE, base.parent / "lib" / "ollama" / EXE]
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "lib" / "ollama" / EXE)
    for path in candidates:
        if path.is_file():
            return path
    raise LlamaServerError("no llama-server found: set TIEBREAK_LLAMA_SERVER, put llama-server on PATH, "
                           "or install an Ollama build that bundles lib/ollama/llama-server")


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """On Windows, put the child in a job object that the OS kills when tiebreak exits, even on a hard kill.

    Without it a crashed or force-stopped tiebreak leaves llama-server holding the model in RAM.
    On other platforms the child is stopped by close() and atexit only.
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class _Limits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class _Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Limits), ("IoInfo", _Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = job and kernel32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    handle = kernel32.OpenProcess(0x0001 | 0x0100, False, proc.pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
    if not (ok and handle and kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(handle))):
        log.warning("could not tie llama-server to tiebreak's lifetime (error %d); "
                    "a hard kill of tiebreak would leave it running", ctypes.get_last_error())
    if handle:
        kernel32.CloseHandle(wintypes.HANDLE(handle))
    # the job handle stays open for the life of this process on purpose: closing it kills the child
    _JOBS.append(job)


_JOBS: list = []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def prob_to_logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p) - math.log1p(-p)


class LlamaServerReranker:
    """Same interface as engine.Reranker: model_id, max_length, score()."""

    # A detokenized prefix can retokenize a few tokens longer; keep room for it.
    ROUNDTRIP_MARGIN = 8

    def __init__(self, name: str, blob: Path, max_length: int = 512, start_timeout_s: int = 120,
                 threads: int | None = None, binary: Path | None = None):
        info = read_header(blob)
        check_reranker(info, name)
        self.model_id = name
        self.blob = blob
        self.arch = info.arch
        self.template = info.rerank_template
        self.softmax_scores = info.arch in SOFTMAX_ARCHS
        self.max_length = min(max_length, info.context_length or max_length)
        self.binary = binary or find_llama_server()
        self.port = _free_port()
        args = [str(self.binary), "-m", str(blob), "--reranking", "--pooling", "rank",
                "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.max_length), "-b", str(self.max_length), "-ub", str(self.max_length),
                "--parallel", "1", "--no-webui"]
        if threads:
            args += ["-t", str(threads)]
        log.info("starting %s for %s (%s)", self.binary, name, info.arch)
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _kill_with_parent(self.proc)
        self._stderr_tail: list[str] = []
        threading.Thread(target=self._drain, daemon=True).start()
        atexit.register(self.close)
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=httpx.Timeout(30.0, connect=5.0))
        try:
            self._wait_ready(start_timeout_s)
            self._overhead = self._template_overhead()
        except Exception:
            self.close()
            raise

    def _drain(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            self._stderr_tail = (self._stderr_tail + [line.decode("utf-8", "replace").rstrip()])[-20:]

    def _wait_ready(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        last = "no answer yet"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                tail = "\n".join(self._stderr_tail[-8:])
                raise LlamaServerError(f"llama-server exited with code {self.proc.returncode} while loading "
                                       f"{self.model_id}:\n{tail}")
            try:
                status = self.http.get("/health", timeout=2.0).status_code
                if status == 200:
                    return
                last = f"/health answered {status}"  # 503 while the model loads
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}: {exc}"  # port not open yet
            time.sleep(0.25)
        self.close()
        raise LlamaServerError(f"llama-server did not become ready within {timeout_s}s "
                               f"(TIEBREAK_LLAMA_START_TIMEOUT_S); last probe: {last}")

    def _tokenize(self, text: str, parse_special: bool = False) -> list[int]:
        r = self.http.post("/tokenize", json={"content": text, "add_special": False, "parse_special": parse_special})
        r.raise_for_status()
        return r.json()["tokens"]

    def _detokenize(self, tokens: list[int]) -> str:
        r = self.http.post("/detokenize", json={"tokens": tokens})
        r.raise_for_status()
        return r.json()["content"]

    def _template_overhead(self) -> int:
        if self.template is None:
            return 4  # at most BOS, EOS, SEP, EOS around query and document
        bare = self.template.replace("{query}", "").replace("{document}", "")
        return len(self._tokenize(bare, parse_special=True)) + 1

    def score(self, query: str, documents: list[str], max_total_tokens: int | None = None,
              deadline: float | None = None) -> Scored:
        q_len = len(self._tokenize(query))
        budget = self.max_length - self._overhead - q_len - self.ROUNDTRIP_MARGIN
        if budget < 1:
            raise LimitError("query", self.max_length,
                             f"query is {q_len} tokens and leaves no room for a document in the "
                             f"{self.max_length}-token window (TIEBREAK_MAX_LENGTH)")
        doc_tokens = [self._tokenize(d) for d in documents]
        total = sum(self._overhead + q_len + len(t) for t in doc_tokens)
        if max_total_tokens is not None and total > max_total_tokens:
            raise LimitError("documents", max_total_tokens,
                             f"request needs {total} tokens (query + documents), "
                             f"limit TIEBREAK_MAX_TOTAL_TOKENS={max_total_tokens}")
        truncated, fed_docs, fed = [], [], 0
        for i, (doc, toks) in enumerate(zip(documents, doc_tokens)):
            if len(toks) > budget:
                truncated.append(i)
                toks = toks[:budget]
                doc = self._detokenize(toks)
            fed_docs.append(doc)
            fed += self._overhead + q_len + len(toks)
        if truncated:
            log.warning("truncated %d document(s) to fit %d tokens", len(truncated), self.max_length)
        remaining = 300.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceeded("request timeout reached before scoring")
        try:
            r = self.http.post("/v1/rerank", json={"query": query, "documents": fed_docs, "top_n": len(fed_docs)},
                               timeout=httpx.Timeout(remaining, connect=5.0))
        except httpx.TimeoutException:
            raise DeadlineExceeded("request timeout reached while scoring") from None
        if r.status_code != 200:
            raise LlamaServerError(f"llama-server answered {r.status_code}: {r.text[:300]}")
        scores = [math.nan] * len(fed_docs)
        for row in r.json()["results"]:
            scores[row["index"]] = float(row["relevance_score"])
        if any(math.isnan(s) for s in scores):
            raise LlamaServerError("llama-server returned fewer scores than documents")
        if self.softmax_scores:
            # Qwen3-style heads end in a softmax over (yes, no), so the server returns P(yes).
            # logit(P(yes)) = logit_yes - logit_no, the quantity the reference implementation ranks by.
            scores = [prob_to_logit(s) for s in scores]
        return Scored(scores=scores, truncated=truncated, tokens=fed)

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        http = getattr(self, "http", None)
        if http is not None:
            http.close()
