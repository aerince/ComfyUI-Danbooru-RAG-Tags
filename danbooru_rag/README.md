# Danbooru RAG module

This module has its own runtime helper, configuration, database, index and
embedding model. It does not import the original Danbooru tag-sorter plugin.

In ComfyUI create `Aerince Danbooru RAG Tags (自然语言转标签)` under
`Aerince/Prompt`, enter a description, and connect `Tags` to the positive
`CLIPTextEncode` text input. The other outputs show the processed description,
JSON diagnostics and processing status. `unload_models` defaults to true;
turning it off keeps both local models loaded and consumes additional memory.

Relative configuration paths are based on this module's directory. Copy
`config.example.json` to `config.local.json` and set a compatible chat GGUF
and llama-server path. The local installation already has those configured.
The distributed embedding model cannot replace the separate chat model.

With the same Python that runs ComfyUI, run this module's
`scripts/index.py --check` for a read-only index check. `--build` explicitly
creates/resumes an index. The supplied index is complete; do not rebuild or
reimport Chinese translations for a normal directory move.

Restart ComfyUI after installing/updating Python code or changing config.
Create a new node or update paths stored in existing workflow widgets.
The legacy node ID is retained as an alias when no old plugin registers it.

The independent distribution has README.md and docs/USAGE.zh-CN.md at its
project root. Install either the Aerince_Tools module or that independent
package, not both. See THIRD_PARTY.md for code, data and model attribution.
