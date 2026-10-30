# v1.0.0

Independent ComfyUI natural-language to Danbooru tag node. This package is not
Aerince_Tools and does not require ComfyUI-Danbooru-Tag-Sorter-Node.

- Faithful conversion and optional description expansion using a local GGUF chat model.
- Cached Excel reads and batched candidate selection.
- 221,783 deduplicated records with existing Chinese translations.
- Prebuilt 1024-dimensional float16 index, portable across installation paths.
- Qwen3-Embedding-0.6B-Q8_0.gguf included.
- Local configuration example, index checker/builder and Chinese usage guide.

Download `ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip` for the complete data and
embedding assets. GitHub's automatic source archives do not include these files.
Supply your own compatible chat GGUF and llama.cpp runtime. No 9B model, private
configuration, CUDA DLLs or other llama.cpp binaries are bundled.

Install either this package or Aerince_Tools with its RAG module, never both.
Read README.md and docs/USAGE.zh-CN.md before installation. Third-party model
and translation/data licenses are retained in danbooru_rag/licenses and
danbooru_rag/THIRD_PARTY.md; the code's MIT license does not relicense them.
