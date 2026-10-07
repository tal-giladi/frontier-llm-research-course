"""Main path: a vLLM 0.30.0 rollout server colocated with the trainer on one GPU (lesson 14.3).

Not run in this build; part of the Module 14 pilot. The control plane follows vLLM v0.30.0's
``examples/rl/rlhf_http_ipc.py`` (read at tag v0.30.0 on 2026-10-07): the server is started with
``--weight-transfer-config '{"backend": "ipc"}'`` and ``VLLM_SERVER_DEV_MODE=1`` (exposes ``/pause`` and
``/resume``), the trainer creates ``WeightTransferTrainerFactory.trainer_init(init_info=IPCTrainerInitInfo(rank=0,
packed=False), client=HTTPVLLMWeightSyncClient(url), source=ModuleSource(model))`` and calls ``send_weights()``
between ``/pause`` and ``/resume``. CUDA IPC needs both processes on the same GPU and
``VLLM_ALLOW_INSECURE_SERIALIZATION=1``. ``VLLM_BATCH_INVARIANT=1`` turns on vLLM's batch-invariant kernels
(beta in v0.30.0, ``docs/features/batch_invariance.md``), which lesson 14.3 compares against the default.

Rollouts use the OpenAI-compatible completions endpoint with token-id prompts, ``n`` samples per prompt,
``logprobs=0`` (the sampled token's log-probability only) and ``return_tokens_as_token_ids`` so that the
trainer can recompute log-probabilities on exactly the tokens the engine sampled. Pilot check: confirm that
the returned log-probabilities are of the *processed* distribution at the requested temperature (vLLM's
``logprobs_mode`` setting) before treating them as the sampler's numbers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


class VLLMServer:
    def __init__(self, model: str, revision: str, port: int = 8000, gpu_memory_utilization: float = 0.35,
                 batch_invariant: bool = False, max_model_len: int = 2048, seed: int = 0):
        self.url = f"http://localhost:{port}"
        self.model = model
        env = os.environ.copy()
        env.update({"VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
        if batch_invariant:
            env["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        args = ["vllm", "serve", model, "--revision", revision, "--dtype", "bfloat16", "--port", str(port),
                "--gpu-memory-utilization", str(gpu_memory_utilization), "--max-model-len", str(max_model_len),
                "--seed", str(seed), "--weight-transfer-config", '{"backend": "ipc"}']
        self.proc = subprocess.Popen(args, env=env, stdout=sys.stdout, stderr=sys.stderr, start_new_session=True)
        self._wait()
        self.engine = None

    def _wait(self, timeout: float = 900):
        import requests
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("vLLM server exited before becoming ready")
            try:
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("vLLM server did not start in time")
            time.sleep(2)

    def attach_trainer(self, model):
        from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient, ModuleSource, WeightTransferTrainerFactory
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
        self.engine = WeightTransferTrainerFactory.trainer_init(
            init_info=IPCTrainerInitInfo(rank=0, packed=False), client=HTTPVLLMWeightSyncClient(self.url),
            source=ModuleSource(model))

    def sync_weights(self):
        """Pause generation, copy the trainer's weights into the engine over CUDA IPC, resume."""
        import requests
        requests.post(f"{self.url}/pause", timeout=60).raise_for_status()
        self.engine.send_weights()
        requests.post(f"{self.url}/resume", timeout=60).raise_for_status()

    def generate(self, prompt_ids: list[list[int]], n: int, max_tokens: int, temperature: float, stop=None):
        """Returns, per prompt, ``n`` tuples (token ids, sampled-token log-probs, finished_with_eos)."""
        import requests
        body = {"model": self.model, "prompt": prompt_ids, "n": n, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": 1.0, "top_k": -1, "logprobs": 0,
                "return_tokens_as_token_ids": True}
        if stop:
            body["stop"] = stop
        r = requests.post(f"{self.url}/v1/completions", json=body, timeout=3600)
        r.raise_for_status()
        choices = sorted(r.json()["choices"], key=lambda c: c["index"])
        out = []
        for c in choices:
            lp = c["logprobs"]
            ids = [int(t.split(":")[1]) for t in lp["tokens"]]
            out.append((ids, [float(x) for x in lp["token_logprobs"]], c.get("finish_reason") == "stop"))
        return [out[i * n:(i + 1) * n] for i in range(len(prompt_ids))]

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
