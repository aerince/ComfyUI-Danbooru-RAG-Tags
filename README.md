# ComfyUI Danbooru RAG Tags

通过本地向量检索与LLM智能匹配，将任意格式提示词转换为结构化 Danbooru 提示词，支持中文，类似tags的短句结构描述匹配效果较好。

- 节点名称：`Aerince Danbooru RAG Tags (自然语言转标签)`
- 分类菜单：`Aerince/Prompt`
- 支持特性：精准概念匹配、单批次批量裁决、开箱即用 22 万+中英双语词库、预构建向量索引。

---

## 快速安装

1. **下载解压**：
   在 [Releases](https://github.com/aerince/ComfyUI-Danbooru-RAG-Tags/releases) 下载完整包 `ComfyUI-Danbooru-RAG-Tags-v1.0.0-full.zip`，解压至 `ComfyUI/custom_nodes/ComfyUI-Danbooru-RAG-Tags`。
   > 完整包内含 22 万标签库、预构建向量索引及 Embedding 模型（约 1GB），解压即用，无需耗时重建。

2. **安装依赖**：
   使用 ComfyUI 的 Python 环境安装依赖：
   ```bash
   python -m pip install -r requirements.txt
   ```
   *(便携包请使用 `python_embeded\python.exe`)*

3. **配置模型与运行器**：
   - 准备 `llama-server`（从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 获取）。
   - 准备一个 GGUF 格式的本地对话模型（如 Qwen3.5-9B）。
   - 复制 `danbooru_rag/config.example.json` 为 `danbooru_rag/config.local.json`，填入路径：
     ```json
     {
       "model_path": "E:/AI/models/LLM/Qwen3.5-9B.gguf",
       "llama_server_path": "D:/Program Files/Llama.CPP/llama-server.exe"
     }
     ```

4. **重启 ComfyUI** 即可在节点列表中找到该节点。

---

## 节点说明

### 节点连接
将输出的 `Tags` 直接连入正向 `CLIPTextEncode` 的 `text` 输入端口即可。

### 核心参数
- **mode**：
  - `忠实转换`：严格提取输入文本中的概念进行标签匹配（推荐日常使用）。
  - `润色扩写后转换`：由模型先对画面细节、构图等进行扩写后再匹配标签。
- **top_k**：每个概念召回的候选标签数量（默认 8）。
- **max_tags**：最终输出标签上限（默认 40）。
- **unload_models**：
  - `执行后卸载`（默认）：生成后释放显存，与后续生图模型不抢显存。
  - `保持加载`：常驻后台，连续转换速度极快。

### 输出端口
- `Tags`：最终合并格式化好的 Danbooru Tag 字符串。
- `扩写后的描述`：扩写模式下的自然语言文案（忠实模式下为原文本）。
- `检索候选`：JSON 格式的候选标签列表，方便排查召回效果。
- `处理信息`：各阶段耗时、命中状态与统计信息。

更详细的高级配置与参数说明，请参考 [完整使用说明](docs/USAGE.zh-CN.md)。

---

## 常见问题

- **Q: 为什么首次转换需要花时间？**
  A: 首次调用时需将 22 万行词库加载至内存（约数秒），后续在相同进程中自动使用内存缓存，瞬间完成。
- **Q: 搬迁文件夹或换盘符需要重建向量索引吗？**
  A: 不需要。索引采用基于模型与内容 SHA-256 的便携签名，迁移后直接识别复用。

---

## 开源协议与致谢

- 代码采用 MIT 协议开源。
- 基础架构与节点逻辑参考自 [ComfyUI-Danbooru-Tag-Sorter-Node](https://github.com/RafealaSilva/ComfyUI-Danbooru-Tag-Sorter-Node) (MIT)。
- 词库中文释义整合自 [WeiLin-Comfyui-Tools](https://github.com/weilin9999/WeiLin-Comfyui-Tools) (GPL-2.0)。
- 向量模型采用阿里 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache-2.0)。
