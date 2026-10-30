import sys

from .rag_tags import DanbooruRAGTagsNode

NODE_CLASS_MAPPINGS = {"AerinceDanbooruRAGTags": DanbooruRAGTagsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"AerinceDanbooruRAGTags": "Aerince Danbooru RAG Tags (自然语言转标签)"}

# Older workflows keep working after the old plugin's RAG registration is removed.
if "DanbooruRAGTagsNode" not in getattr(sys.modules.get("nodes"), "NODE_CLASS_MAPPINGS", {}):
    NODE_CLASS_MAPPINGS["DanbooruRAGTagsNode"] = DanbooruRAGTagsNode
    NODE_DISPLAY_NAME_MAPPINGS["DanbooruRAGTagsNode"] = "Danbooru RAG Tags (Aerince migration)"
