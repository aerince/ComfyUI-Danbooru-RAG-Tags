# ComfyUI-Danbooru-RAG-Tags

将中文或英文自然语言转换为 Danbooru 标签的独立 ComfyUI RAG 节点。先从本地词库检索，再由本地模型选择标签；支持忠实转换和润色扩写。

- 新节点：`Aerince Danbooru RAG Tags (自然语言转标签)`
- 菜单分类：`Aerince/Prompt`
- 新工作流的 `class_type`：`AerinceDanbooruRAGTags`
- 环境：ComfyUI 现有 Python，要求 Python >= 3.11。

本项目独立分发，不是 `Aerince_Tools` 安装包。`Aerince_Tools` 中也有相同模块，**不要同时安装两种**。

## 下载与安装

发布时请在 [Releases](https://github.com/aerince/ComfyUI-Danbooru-RAG-Tags/releases) 下载 `ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip`。GitHub 自动生成的 `Source code` 不含大文件，不能替代完整包。

完整包包含源码、Excel 词库、预构建向量索引及 `Qwen3-Embedding-0.6B-Q8_0.gguf`。不含 9B 聊天模型、llama.cpp 二进制或私有配置。本文提供发布准备说明，不表示 Release 已创建或上传。

1. 解压至 `ComfyUI/custom_nodes/ComfyUI-Danbooru-RAG-Tags`，确认该目录下直接有根 `__init__.py`，不要多套一层文件夹。
2. 在该目录使用 **ComfyUI 实际使用的 Python** 安装依赖：

   ```console
   python -m pip install -r requirements.txt
   ```

   这里的 `python` 必须指向 ComfyUI 环境；便携版应使用其 `python_embeded` 中的解释器。依赖为 `numpy>=1.24`、`pandas>=2`、`openpyxl>=3.1`。

3. 从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 自行下载适合平台的 CPU 或 CUDA 完整运行包，保留配套 DLL/动态库。可放入 `danbooru_rag/bin`，或配置可执行文件的绝对路径。
4. 复制 `danbooru_rag/config.example.json` 为同目录的 `config.local.json`，设置 `model_path` 为自己的聊天模型 GGUF 路径，设置 `llama_server_path` 为 `llama-server` 可执行文件路径。本地 Qwen3.5-9B 的 JSON schema 路径受支持，模型需自行准备；其他聊天模型尚未验证。
5. 重启 ComfyUI，新建该节点，将 `Tags` 输出连接到正向 `CLIPTextEncode` 的 `text` 输入。

详细步骤、全部参数、四个输出、旧工作流迁移和故障排查见 [中文使用说明](docs/USAGE.zh-CN.md)。

## 数据与迁移

词库含 221,788 条原始行、221,783 条有效去重记录，已包含 Weilin 中文翻译，不需要再次导入。词库可能包含成人标签，请按使用场景筛选。

完整包的资源位于 `danbooru_rag` 下：

```text
danbooru_rag/
  data/danbooru_tags.xlsx
  data/rag_index/<portable-sha>/state.json
  data/rag_index/<portable-sha>/vectors.npy
  models/Qwen3-Embedding-0.6B-Q8_0.gguf
```

向量为 1024 维 float16；`vectors.npy` 为 454,211,712 字节，embedding 模型为 639,150,592 字节。索引签名取决于模型内容 SHA-256 和记录内容，不包含绝对路径或 mtime，完整搬迁无需重建。中文、分类、记录顺序或 embedding 模型内容改变时会使用新的索引。

旧工作流的 `DanbooruRAGTagsNode` 保留条件兼容注册：旧原插件尚未注册该 ID 时才注册兼容别名。新工作流请使用 `AerinceDanbooruRAGTags`；不要同时加载本项目和 `Aerince_Tools`。

## 发布准备

源码 Git 忽略数据、模型、`config.local.json`、`bin` 和 `output`；仅克隆源码需要另行补齐资源。完整分发 ZIP 的本地目标路径为：

```text
output/ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip
```

本地仓库已准备好源码提交。确认 GitHub CLI 已登录后，在项目目录先查看发布操作（不会联网写入）：

```powershell
pwsh -NoProfile -File scripts/publish.ps1
```

实际创建仓库、推送源码并上传完整包（默认新仓库为私有；已有仓库可见性不变）：

```powershell
pwsh -NoProfile -File scripts/publish.ps1 -Publish
```

需要新建公开仓库时，在命令末尾加 `-Public`。脚本不会自动覆盖已有 Release；中途失败会直接停止。以上仅为待执行命令，不表示已执行外部发布。

## 授权与来源

- 代码沿用 MIT，保留 `Copyright (c) 2025 RafealaSilva`，并加入 aerince 修改署名。
- 数据来自原插件 Excel 及 Weilin 中文翻译，不是本项目作者原创，也不因代码采用 MIT 而统一变为 MIT 数据。Weilin 的 GPL v2 许可随包位于 `danbooru_rag/licenses/Weilin-GPL-2.0.txt`。
- embedding 模型来自官方 [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)，采用 Apache 2.0；许可随包位于 `danbooru_rag/licenses/Qwen-Apache-2.0.txt`。
- 自备聊天模型及 llama.cpp 运行包遵循各自许可。上述代码许可不覆盖完整分发包中的全部第三方内容。
