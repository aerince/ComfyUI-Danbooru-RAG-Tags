"""Node-owned llama.cpp subprocess lifecycle (derived from the MIT tag sorter)."""
from collections import deque
import json
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import comfy.model_management as model_management

_LOCK = threading.RLock()


def _request_json(url, api_key, payload=None, timeout=2):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=headers)
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        with error:
            detail = error.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp HTTP {error.code}: {detail}".replace(api_key, "[redacted]")) from None


class _LocalLlama:
    def __init__(self, executable, model, gpu_layers, timeout, embedding=False):
        self.executable = Path(executable.strip().strip('\"\'')).resolve()
        self.model = Path(model.strip().strip('\"\'')).resolve()
        self.gpu_layers = int(gpu_layers)
        self.embedding = embedding
        self.deadline = time.monotonic() + int(timeout)
        self.process = None
        self.reader = None
        self.tail = deque(maxlen=40)
        self.api_key = secrets.token_urlsafe(24)

    def _read_logs(self):
        for line in self.process.stdout:
            self.tail.append(line.rstrip())

    def _diagnostic(self):
        return "\n".join(self.tail)[-4000:].replace(self.api_key, "[redacted]")

    def __enter__(self):
        try:
            for path in (self.executable, self.model):
                if not path.is_file():
                    raise FileNotFoundError(f"找不到本地运行器或模型：{path}")
            model_management.throw_exception_if_processing_interrupted()
            if self.gpu_layers != 0:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            self.base_url = f"http://127.0.0.1:{port}"
            command = [str(self.executable), "--model", str(self.model),
                       "--host", "127.0.0.1", "--port", str(port),
                       "--ctx-size", "8192", "--n-gpu-layers", str(self.gpu_layers),
                       "--parallel", "1", "--alias", "danbooru-tag-classifier", "--api-key", self.api_key]
            if self.embedding:
                command.extend(["--embedding", "--pooling", "last", "--batch-size", "2048", "--ubatch-size", "2048"])
            else:
                command.extend(["--jinja", "--reasoning", "off"])
            self.process = subprocess.Popen(
                command, cwd=str(self.executable.parent), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.reader = threading.Thread(target=self._read_logs, daemon=True)
            self.reader.start()
            while time.monotonic() < self.deadline:
                model_management.throw_exception_if_processing_interrupted()
                if self.process.poll() is not None:
                    raise RuntimeError("本地模型加载失败：\n" + self._diagnostic())
                try:
                    if _request_json(self.base_url + "/health", self.api_key, timeout=1).get("status") == "ok":
                        return self
                except (RuntimeError, URLError, OSError, ValueError):
                    pass
                time.sleep(0.1)
            raise TimeoutError("本地模型加载超时。\n" + self._diagnostic())
        except BaseException:
            self.close()
            raise

    def chat(self, payload):
        return self._call("/v1/chat/completions", payload)

    def embeddings(self, texts):
        return self._call("/v1/embeddings", {
            "model": "danbooru-tag-classifier", "input": texts, "encoding_format": "float"})

    def _call(self, endpoint, payload):
        done = threading.Event()
        result = {}

        def request():
            try:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("本地推理超时。")
                result["data"] = _request_json(self.base_url + endpoint, self.api_key, payload, timeout=remaining)
            except Exception as error:
                result["error"] = error
            finally:
                done.set()

        threading.Thread(target=request, daemon=True).start()
        while not done.wait(0.1):
            model_management.throw_exception_if_processing_interrupted()
            if time.monotonic() >= self.deadline:
                raise TimeoutError("本地推理超时。")
            if self.process.poll() is not None:
                raise RuntimeError("本地模型意外退出：\n" + self._diagnostic())
        model_management.throw_exception_if_processing_interrupted()
        if "error" in result:
            raise result["error"]
        return result["data"]

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.reader is not None:
                self.reader.join(timeout=1)
            if self.process.stdout is not None:
                self.process.stdout.close()

    def __exit__(self, *_):
        self.close()
