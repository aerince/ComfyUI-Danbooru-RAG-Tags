"""Check the packaged index; --build explicitly creates or resumes vectors."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "danbooru_rag").is_dir():
    ROOT = ROOT / "danbooru_rag"


class ConsoleProgress:
    def __init__(self, total):
        self.total = total
        self.last = -5000

    def update_absolute(self, value):
        if value - self.last >= 5000 or value == self.total:
            print(f"INDEX_PROGRESS {value}/{self.total}", flush=True)
            self.last = value


def load_rag():
    # Standalone use has no ComfyUI GPU model to unload or host interrupt to poll.
    comfy = types.ModuleType("comfy")
    management = types.ModuleType("comfy.model_management")
    management.InterruptProcessingException = type("InterruptProcessingException", (Exception,), {})
    management.throw_exception_if_processing_interrupted = lambda: None
    management.unload_all_models = lambda: None
    management.soft_empty_cache = lambda: None
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = ConsoleProgress
    comfy.model_management = management
    comfy.utils = utils
    sys.modules.update({"comfy": comfy, "comfy.model_management": management, "comfy.utils": utils})
    package = types.ModuleType("aerince_rag_index_cli")
    package.__path__ = [str(ROOT)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".rag_tags", ROOT / "rag_tags.py")
    rag = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rag
    spec.loader.exec_module(rag)
    return rag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only check (default)")
    mode.add_argument("--build", action="store_true", help="Explicitly build/resume the index")
    parser.add_argument("--batch-size", type=int, default=64, choices=range(1, 65), metavar="1..64")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    args = parser.parse_args()
    rag = load_rag()
    database = Path(rag.resolve_database_path())
    model = Path(rag.DEFAULT_EMBEDDING)
    for path in (database, model):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = rag._read_records(database)
    directory = rag.INDEX_ROOT / rag._signature(database, model, records)
    matrix, state = rag._load_index(directory, len(records))
    complete = matrix is not None and state["completed"] == len(records)
    del matrix
    print(json.dumps({"database": str(database), "embedding_model": str(model),
                      "records": len(records), "index": str(directory), "state": state,
                      "already_complete": complete, "build_requested": args.build,
                      "chat_model_available": Path(rag.DEFAULT_MODEL).is_file(),
                      "llama_server_available": Path(rag.DEFAULT_SERVER).is_file()},
                     ensure_ascii=False, indent=2), flush=True)
    if not args.build:
        if not complete:
            raise RuntimeError("No complete index for these records and embedding model; use --build explicitly.")
        print("CHECK_ONLY_OK: complete index; no models launched or index files written.")
        return
    if complete:
        print("INDEX_ALREADY_COMPLETE")
        return
    started = time.monotonic()
    with rag._LocalLlama(rag.DEFAULT_SERVER, str(model), args.gpu_layers, 86400, embedding=True) as server:
        matrix = rag._build_index(server, directory, records, args.batch_size)
    del matrix
    matrix, state = rag._load_index(directory, len(records))
    if matrix is None or state["completed"] != len(records):
        raise RuntimeError("The built index did not reload as complete")
    print("INDEX_COMPLETE", matrix.shape, "seconds", round(time.monotonic() - started, 1))


if __name__ == "__main__":
    main()
