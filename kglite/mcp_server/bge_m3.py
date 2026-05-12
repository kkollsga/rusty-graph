"""BAAI/bge-m3 embedder via direct ONNX Runtime + HuggingFace Hub.

Bypasses `fastembed-python` because its model catalog doesn't include
`BAAI/bge-m3` (only the Rust `fastembed-rs` crate carries the BGEM3
variant). 0.9.20 hit this gap and broke text_score on every
embedder-enabled deployment.

Same files fastembed-rs uses: `onnx/model.onnx` + the external data
shard `onnx/model.onnx_data` (the model is too large for a single
flatbuffer) + `onnx/tokenizer.json`. Runs via `onnxruntime.InferenceSession`,
tokenises via `tokenizers.Tokenizer.from_file`, applies CLS pooling
(matches fastembed-rs's default `Pooling::Cls` for BGEM3). No L2
normalisation by default — same as fastembed-rs.

Cache lives at `~/.cache/fastembed/BAAI--bge-m3/` so the operator's
existing 0.9.18-downloaded weights are reused without re-download.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading

log = logging.getLogger("kglite.mcp_server.bge_m3")

MODEL_ID = "BAAI/bge-m3"
DIMENSION = 1024
MAX_LENGTH = 8192  # bge-m3 model_max_length


class BgeM3Embedder:
    """Duck-typed Embedder for kglite. Same `dimension` + `embed` +
    `load` + `unload` shape kglite's `g.set_embedder(model)` accepts."""

    dimension = DIMENSION

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path(
                os.environ.get(
                    "FASTEMBED_CACHE_PATH",
                    Path.home() / ".cache" / "fastembed",
                )
            )
        # fastembed lays things out as `<cache>/models--BAAI--bge-m3/`;
        # we use the same dir so downloaded weights are shared across
        # fastembed-rs (0.9.18), fastembed-python (other models), and us.
        self._cache_dir = Path(cache_dir)
        self._lock = threading.Lock()
        self._session = None  # ort.InferenceSession when loaded
        self._tokenizer = None  # tokenizers.Tokenizer when loaded
        self._input_names: list[str] = []

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from tokenizers import Tokenizer

            # Materialise model + external data + tokenizer into the
            # shared cache. hf_hub_download is idempotent — re-running
            # with cached files is essentially free.
            model_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename="onnx/model.onnx",
                cache_dir=str(self._cache_dir),
            )
            # External weights shard must be next to the .onnx file so
            # onnxruntime resolves the relative path inside model.onnx.
            hf_hub_download(
                repo_id=MODEL_ID,
                filename="onnx/model.onnx_data",
                cache_dir=str(self._cache_dir),
            )
            tokenizer_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename="onnx/tokenizer.json",
                cache_dir=str(self._cache_dir),
            )

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [i.name for i in session.get_inputs()]
            tokenizer = Tokenizer.from_file(tokenizer_path)
            # bge-m3 input is 8192-cap; truncate the tokenizer to match.
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

            self._session = session
            self._tokenizer = tokenizer
            log.info("bge-m3 loaded from cache (%s)", self._cache_dir)

    def unload(self) -> None:
        """Drop the ORT session + tokenizer so the ~2 GB of weights
        can be freed between idle periods. Re-materialised from disk
        on next `load()` — no re-download."""
        with self._lock:
            self._session = None
            self._tokenizer = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.load()
        import numpy as np

        with self._lock:
            tokenizer = self._tokenizer
            session = self._session
            input_names = list(self._input_names)
        assert tokenizer is not None and session is not None

        encoded = tokenizer.encode_batch(list(texts))
        max_len = max((len(e.ids) for e in encoded), default=1)
        input_ids = np.array([e.ids + [1] * (max_len - len(e.ids)) for e in encoded], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encoded],
            dtype=np.int64,
        )

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = session.run(None, feeds)
        # bge-m3 ONNX export has a single output: last_hidden_state
        # shape (batch, seq, hidden_dim). CLS pool = take token 0.
        last_hidden = outputs[0]
        embeddings = last_hidden[:, 0, :]
        return embeddings.tolist()
