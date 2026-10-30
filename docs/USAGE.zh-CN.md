# 节点使用指南

## 1. 快速上手

1. 下载并解压完整分发包到 `ComfyUI/custom_nodes/ComfyUI-Danbooru-RAG-Tags`。
2. 安装依赖：
   ```bash
   python -m pip install -r requirements.txt
   ```
3. 复制 `danbooru_rag/config.example.json` 为 `danbooru_rag/config.local.json`，配置本地 `model_path` 和 `llama_server_path`。
4. 重启 ComfyUI，在 `Aerince/Prompt` 下找到节点。

---

## 2. 参数说明

| 参数名 | 默认值 | 说明 |
| --- | --- | --- |
| `description` | 空 | 输入你的画面描述（支持中英混输，例如：“1个女孩，黑发，坐在长椅上，看向镜头”）。 |
| `mode` | `忠实转换` | **忠实转换**：按用户原意直接转写。<br>**润色扩写后转换**：先让模型对画面细节进行丰富扩写，再提取标签。 |
| `top_k` | `8` | 向量库为每个概念召回的候选词数量。 |
| `max_tags` | `40` | 最终生成的最大标签数量上限。 |
| `unload_models` | `True` | **执行后卸载**：转写完成后自动释放显存，避免影响后续出图。<br>**保持加载**：模型常驻后台，连续转换速度更快。 |
| `excel_file` | 默认内置 | 标签数据库文件。留空即使用内置的 22 万条中英词库。 |
| `gpu_layers` | `-1` | 聊天模型 GPU 卸载层数（-1 为全量 GPU，0 为纯 CPU 运行）。 |
| `embedding_gpu_layers` | `-1` | Embedding 模型 GPU 卸载层数。 |

---

## 3. 输出接口

- **Tags**（文本）：排版整理好的英文 Danbooru 标签，直接连入正向提示词 `CLIPTextEncode`。
- **扩写后的描述**（文本）：展示模型扩写后的细节文案，方便核对。
- **检索候选**（JSON 文本）：展示每个概念召回并参与裁决的全部候选标签。
- **处理信息**（文本）：显示耗时、索引命中状态、匹配词数等。

---

## 4. 常见排错

- **找不到模型或 llama-server**：请检查 `danbooru_rag/config.local.json` 中的路径是否正确填入。
- **显存紧张或爆显存**：将 `unload_models` 保持开启（默认），或将 `gpu_layers` 改为 `0` 使用 CPU 运算。
- **换盘符后是否需要重建索引**：不需要。向量索引与文件物理路径解耦，完整移动目录后会自动命中已有向量。
