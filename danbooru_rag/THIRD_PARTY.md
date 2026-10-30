# Third-party notices

## Code

The local llama.cpp process helper and the initial tag database originate from
[RafealaSilva/ComfyUI-Danbooru-Tag-Sorter-Node](https://github.com/RafealaSilva/ComfyUI-Danbooru-Tag-Sorter-Node).
The original MIT copyright notice is retained in `LICENSE`.

Aerince's RAG conversion, caching, batch selection, portable path configuration
and packaging changes were made in 2026. They are not an upstream release.
Code in this module is distributed under the MIT license. This does not
relicense any bundled third-party data or model weights.

## Database and prebuilt vectors

`data/danbooru_tags.xlsx` is the existing tag-sorter workbook, with Chinese
translations partially replaced from
[weilin9999/WeiLin-Comfyui-Tools](https://github.com/weilin9999/WeiLin-Comfyui-Tools).
The latter repository carries GPL version 2; a verbatim copy is provided as
`licenses/Weilin-GPL-2.0.txt`. Its translation contribution and the derived
translation-containing database are distributed under those terms, not as
new MIT-only data. The editable workbook is included in the full package.
Underlying tag names and upstream data retain their applicable rights.

Modification date: 2026-10-30. Only existing tag translations were replaced;
38,518 Chinese cells changed. No tags, categories, subcategories or workbook
row order were added/removed by the translation import. There are 221,788
source rows and 221,783 effective deduplicated records. The prebuilt vectors
derive from these records using complete English tag plus Chinese text.
Category fields are retrieval metadata and are not embedding text.

The migration reuses the finished vectors; it does not repeat the translation
import. The original source repositories are linked above. No upstream
endorsement or exclusive authorship of this dataset is claimed.

## Embedding model

`models/Qwen3-Embedding-0.6B-Q8_0.gguf` is an unchanged local copy of the
Qwen3-Embedding-0.6B GGUF Q8_0 model, attributed to the Qwen team / Alibaba Cloud.

- Model: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF
- Base model: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- License: Apache License 2.0, included as `licenses/Qwen-Apache-2.0.txt`.
- Size: 639,150,592 bytes. Output used here: 1024 dimensions.

No 9B chat-model weights or llama.cpp executables are included in the
distribution. Users obtain those separately under their respective licenses.
