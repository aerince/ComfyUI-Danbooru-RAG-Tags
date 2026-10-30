# 中文使用说明

## 1. 安装与资源

本项目是独立的 ComfyUI 自然语言转 Danbooru 标签 RAG 节点，不属于 `Aerince_Tools` 安装包。二者含相同模块，选择一种安装，不要同时加载。

发布时请在 [Releases](https://github.com/aerince/ComfyUI-Danbooru-RAG-Tags/releases) 下载 `ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip`。GitHub 自动生成的 `Source code` 不含大文件。本文不表示 Release 已发布或附件已上传。

解压后的节点根目录应为 `ComfyUI/custom_nodes/ComfyUI-Danbooru-RAG-Tags`，其中直接包含根 `__init__.py`、`requirements.txt` 及 `danbooru_rag` 目录。主模块为 `danbooru_rag/{__init__,config,runtime,rag_tags}.py`，根 `__init__.py` 导出节点。

完整包包含：

| 资源 | 相对节点根目录的路径 | 说明 |
| --- | --- | --- |
| 词库 | `danbooru_rag/data/danbooru_tags.xlsx` | 221,788 原始行，221,783 有效去重记录，含 Weilin 中文翻译 |
| 索引状态 | `danbooru_rag/data/rag_index/<portable-sha>/state.json` | 内容签名对应的索引状态 |
| 向量 | `danbooru_rag/data/rag_index/<portable-sha>/vectors.npy` | 1024 维 float16，454,211,712 字节 |
| embedding 模型 | `danbooru_rag/models/Qwen3-Embedding-0.6B-Q8_0.gguf` | 639,150,592 字节 |

`<portable-sha>` 表示实际的便携签名目录名，不需要手工创建这个字面名称。完整包不包含 9B 聊天模型、llama.cpp 二进制或私有配置。

### 安装 Python 依赖

要求 Python >= 3.11。进入节点根目录，使用 **运行 ComfyUI 的同一个 Python**：

```console
python -m pip install -r requirements.txt
```

依赖为 `numpy>=1.24`、`pandas>=2`、`openpyxl>=3.1`。终端中的 `python` 若不是 ComfyUI 的解释器，请用实际解释器替换命令开头。Windows 便携版应使用发行包中 `python_embeded/python.exe`，虚拟环境则使用该环境的 Python，不要安装到无关的系统 Python。

### 安装 llama.cpp 与聊天模型

1. 在 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载适合操作系统与硬件的 CPU 或 CUDA 完整运行包。
2. 完整解压并保留配套 DLL/动态库，不能只复制 `llama-server.exe`。可将运行包放入 `danbooru_rag/bin`，也可放在其他位置并使用绝对路径。
3. 自行准备聊天模型 GGUF。本地 Qwen3.5-9B 的 JSON schema 路径受支持；其他模型尚未验证。完整分发包中的 Qwen3-Embedding-0.6B 是向量模型，不能替代聊天模型。
4. 配置完成后重启 ComfyUI。

## 2. 配置与路径

将 `danbooru_rag/config.example.json` 复制为同目录的 `config.local.json`，保留示例配置结构，主要设置：

| 配置项 | 用途 |
| --- | --- |
| `model_path` | 自备聊天模型 GGUF 路径 |
| `embedding_model_path` | embedding GGUF 路径；完整包可使用内置模型 |
| `llama_server_path` | `llama-server.exe` 或 `llama-server` 可执行文件路径 |
| `database_path` | 默认 Excel 词库路径 |
| `index_root` | 向量索引根目录 |

**所有相对路径始终以 `danbooru_rag` 目录为基准**，不以终端当前目录、ComfyUI 根目录或工作流目录为基准。例如 `models/Qwen3-Embedding-0.6B-Q8_0.gguf` 指向本模块的 `models` 目录。绝对路径也受支持。手动编辑 JSON 时，Windows 路径可使用 `/`，或将反斜杠写为 `\\`。

以下环境变量可覆盖对应默认配置：

| 环境变量 | 对应配置 |
| --- | --- |
| `AERINCE_RAG_MODEL_PATH` | `model_path` |
| `AERINCE_RAG_EMBEDDING_MODEL_PATH` | `embedding_model_path` |
| `AERINCE_RAG_LLAMA_SERVER_PATH` | `llama_server_path` |
| `AERINCE_RAG_DATABASE_PATH` | `database_path` |
| `AERINCE_RAG_INDEX_ROOT` | `index_root` |

配置在模块导入时读取。修改配置文件或环境变量后，应重启 ComfyUI 并新建节点，以采用新的默认值。已保存工作流的 widget 可能保留旧绝对路径，必须自行更新，重启不会自动覆盖它们。

节点中的 `excel_file` 填 `danbooru_tags.xlsx` 或留空即可使用默认内置数据库；需要自定义词库时可填写 Excel 文件绝对路径。内置词库已包含中文翻译，不需要再次导入 Weilin 数据。

## 3. 创建节点与连接

在 `Aerince/Prompt` 分类创建 `Aerince Danbooru RAG Tags (自然语言转标签)`。新工作流的 `class_type` 为 `AerinceDanbooruRAGTags`。

最小连接方式：

```text
自然语言描述
    -> Aerince Danbooru RAG Tags (自然语言转标签)
    -> Tags
    -> CLIPTextEncode 的 text 输入（正向 conditioning 分支）
```

输入可以是中文、英文或混合描述。输出标签用于正向提示词，不要将其当作 negative prompt。词库可能含成人标签，运行前后请按实际用途检查内容。

## 4. 模式与参数

### 转换模式

- **忠实模式**：围绕输入中已有概念选词，尽量保持原意。
- **润色扩写模式**：先扩写描述，再检索和选择标签，可能引入额外细节；请核对扩写结果。
- `append_relations` 默认 `False`，只在扩写模式有效。关系补充不保证消除人物位置、动作对象或空间关系误判。

### 参数参考

| 参数 | 默认值或范围 | 用途 |
| --- | --- | --- |
| `excel_file` | 默认显示内置 Excel 的绝对路径；`danbooru_tags.xlsx` 或留空等效 | 默认内置词库；也可填写自定义 Excel 的绝对路径 |
| `top_k` | 默认 8，范围 2-16 | 每个待检索概念的候选数量 |
| `max_tags` | 默认 40，范围 1-80 | 控制输出标签数量上限 |
| `embedding_batch_size` | 默认 32，范围 1-64 | 仅用于建库阶段的 embedding 批量大小 |
| `embedding_gpu_layers` | 默认 -1；0 为 CPU | embedding 模型的 GPU 卸载层数 |
| `gpu_layers` | 默认 -1；0 为 CPU | 聊天模型的 GPU 卸载层数 |
| `timeout` | 默认 7200 秒 | 模型处理的等待上限 |
| `append_relations` | 默认 `False` | 仅扩写模式使用的关系补充 |
| `unload_models` | 默认 `True` | 本次执行后卸载模型；`False` 保留模型供后续使用 |

精确匹配的词条直接通过，剩余概念批量送入选词；只有实际超过上下文容量或输出被截断时才拆分请求。`embedding_batch_size` 不是日常选词请求的拆分开关。增大 `top_k` 会增加候选和上下文需求，不等于一定更准确。

## 5. 四个输出

| 输出 | 内容 | 常见用途 |
| --- | --- | --- |
| `Tags` | 选出的 Danbooru 标签文本 | 连接正向 `CLIPTextEncode` |
| 扩写后的描述 | 处理后的自然语言描述 | 检查扩写是否偏离原意 |
| 检索候选 (JSON) | 检索得到的候选记录 | 检查概念召回及最终选词依据 |
| 处理信息 | 本次处理的状态和说明 | 定位配置、模型及索引相关问题 |

后面三个输出适合连接文本显示节点查看，不需要全部连接才能生成图像。标签检索和模型选择不能保证完整表达复杂关系，尤其是多人交互、左右位置和动作指向；应结合扩写描述与候选输出检查。

## 6. CPU、GPU 与显存

- **仅 CPU**：使用 CPU 运行包，并将 `embedding_gpu_layers` 和 `gpu_layers` 都设为 `0`。无需 GPU，但速度及内存占用取决于模型和设备。
- **GPU**：使用匹配硬件与平台的运行包。两个层数参数默认 `-1`，请求尽可能将模型层卸载到 GPU；可改为非负层数减少显存使用。
- **混合运行**：例如 embedding 使用 CPU，聊天模型使用 GPU，分别设置两个参数即可。
- **执行后释放**：`unload_models=True` 为默认值，适合与 ComfyUI 图像模型共享显存。
- **连续处理**：`unload_models=False` 会保留两类模型；启用 GPU 卸载时它们会继续占用显存，可能与后续图像生成争用资源。

显存不足时先使用默认卸载设置，再降低 GPU 卸载层数或改用 CPU。若问题出现在建库阶段，可降低 `embedding_batch_size`。

## 7. 索引检查、续建与迁移

完整包携带预构建索引，数据和 embedding 模型内容未改动时，通常直接使用，不需要重新计算全部向量。

在节点根目录，使用 ComfyUI Python 运行只读检查：

```console
python scripts/index.py --check
```

检查不会创建或补写索引。需要构建或从已有进度继续构建时，显式执行：

```console
python scripts/index.py --build
```

建库需要正确配置 embedding 模型、llama.cpp 及资源路径。不要将 `--build` 当作每次启动前的必做步骤。完整包中的索引目录应连同 `state.json` 和 `vectors.npy` 一起保留，不能只迁移其中一个文件。

索引签名基于 **embedding 模型内容 SHA-256 + 记录内容**，不含绝对路径或 mtime：

- 整体搬迁目录、换盘符或文件时间变化，内容不变时无需重建。
- 中文翻译、分类、记录顺序或 embedding 模型内容改变时，会使用不同的索引签名，需要为新内容构建索引。
- 仅调整提示词、`top_k` 或 `max_tags` 不属于词库或 embedding 内容变化。

词库缓存会在同一进程中根据 Excel 文件大小和 mtime 自动刷新；这是词库读取缓存的刷新机制，不是便携索引签名。配置默认值变更仍需重启。首次词库加载约 10 秒仅为旧环境测试参考，不是本版本或当前设备的新测试结果。

迁移时复制完整资源，并修改指向外部聊天模型或 llama.cpp 的路径。模型与数据未变时不要因目录迁移主动重建索引。

## 8. 旧工作流迁移

旧 `class_type` 为 `DanbooruRAGTagsNode`。本项目只在已安装的旧原插件尚未注册该 ID 时注册兼容别名，避免覆盖旧插件的节点定义；该别名不能保证与旧插件同时安装时接管已有工作流。

迁移步骤：

1. 保留旧工作流副本，避免直接覆盖唯一版本。
2. 在独立版与 `Aerince_Tools` 之间选择一个，停用另一种安装，再重启 ComfyUI。
3. 在工作流中新建 `Aerince Danbooru RAG Tags (自然语言转标签)`，逐项迁移输入与参数，重新连接四个输出中实际需要的连接。
4. 检查已保存的路径 widget，更新旧机器、旧插件目录或旧盘符的绝对路径。
5. 保存新工作流，确认使用 `AerinceDanbooruRAGTags`。

不建议仅在工作流 JSON 中批量替换旧 ID：新旧 widget 或连接可能不同。兼容别名是帮助旧工作流加载的措施，新流程请明确使用新 ID。

## 9. 故障排查

| 现象 | 首先检查 |
| --- | --- |
| 节点不出现在菜单 | 解压层级是否正确、是否重启、ComfyUI 启动日志是否提示依赖缺失；确认 Python >= 3.11 |
| `numpy`、`pandas` 或 `openpyxl` 导入失败 | 是否用 ComfyUI 的 Python 安装 `requirements.txt`，而不是另一个系统环境 |
| 找不到 Excel、embedding 模型或预建索引 | 是否下载完整 ZIP；GitHub 的自动源码包不带大文件；相对路径以 `danbooru_rag` 为基准 |
| 找不到聊天模型 | 完整包不含 9B 模型；检查 `model_path`、环境变量及旧工作流 widget |
| `llama-server` 启动失败或缺 DLL | 是否下载对应平台的完整运行包、保留配套动态库、配置了正确可执行文件路径 |
| 更改配置后节点仍显示旧路径 | 重启并新建节点；已保存的 widget 值需要手动更新 |
| CUDA 或显存错误 | 检查运行包与硬件；降低卸载层数，或将两个层数参数设为 0 使用 CPU |
| 建库期间内存或显存不足 | 降低 `embedding_batch_size`，检查 embedding 模型的 GPU 层数 |
| 迁移后似乎重新建库 | 先运行只读 `--check`；确认索引目录已完整复制，词库记录及 embedding 模型内容未改变 |
| 请求超时 | 查看处理信息与服务错误，确认模型正常启动；CPU 可能较慢，必要时调整默认 7200 秒超时 |
| 选词失败或 JSON schema 报错 | 先核对聊天模型与 llama.cpp；目前支持本地 Qwen3.5-9B 路径，其他模型尚未验证 |
| 旧流程加载到其他插件的节点 | 旧 ID 是条件注册别名；新建 `AerinceDanbooruRAGTags` 并迁移连接 |
| 标签关系错误或扩写添加了多余内容 | 检查扩写描述和候选 JSON，改用忠实模式或明确人物、位置及动作；关系补充不保证消除误判 |

## 10. 数据与许可

代码沿用 MIT，保留 `Copyright (c) 2025 RafealaSilva`，并加入 aerince 修改署名。数据库来自原插件 Excel 与 Weilin 中文翻译，不是本项目作者原创；**代码的 MIT 不代表所有第三方数据都按 MIT 授权**。Weilin GPL v2 许可位于 `danbooru_rag/licenses/Weilin-GPL-2.0.txt`。

embedding 模型来自官方 [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)，采用 Apache 2.0，许可位于 `danbooru_rag/licenses/Qwen-Apache-2.0.txt`。重新分发时应保留对应来源与许可。自备聊天模型和 llama.cpp 运行包另按其各自许可使用。

本仓库源码 Git 忽略数据、模型、`config.local.json`、`bin` 与 `output`，不要将私有模型路径或本地配置提交到公开仓库。维护者的完整分发包目标为 `output/ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip`，待执行的 GitHub Release 命令见 [README](../README.md#发布准备)。
