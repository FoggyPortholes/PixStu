import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

try:  # huggingface hub is optional at runtime
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency failure handled at runtime
    hf_hub_download = None  # type: ignore


ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = os.path.join(ROOT, "configs")
OUTPUTS = os.path.join(ROOT, "outputs")
GALLERY = os.path.join(ROOT, "reference_gallery")
MODELS = os.path.join(ROOT, "models")
LORAS = os.path.join(MODELS, "lora")
CONTROLNET_DIR = os.path.join(MODELS, "controlnet")
IP_ADAPTER_DIR = os.path.join(MODELS, "ip_adapter")
LOGS = os.path.join(ROOT, "logs")

CONTROLNET_CATALOG_FILE = os.path.join(CONFIGS, "controlnet_catalog.json")
IP_ADAPTER_CATALOG_FILE = os.path.join(CONFIGS, "ip_adapter_catalog.json")

CATALOG_FILE = os.path.join(CONFIGS, "lora_catalog.json")
SUPPORTED_EXTS = (".safetensors", ".ckpt", ".bin", ".pt")


def ensure_directories() -> None:
    for path in [OUTPUTS, GALLERY, LORAS, CONTROLNET_DIR, IP_ADAPTER_DIR, LOGS]:
        os.makedirs(path, exist_ok=True)


def resolve_model_path(*segments: str) -> str:
    return os.path.join(MODELS, *segments)


ensure_directories()


@dataclass
class LoraRecord:
    name: str
    filename: str
    path: str
    local_path: str
    exists: bool
    description: str = ""
    repo_id: Optional[str] = None
    preview_url: Optional[str] = None
    tags: Optional[List[str]] = None

    def to_display(self) -> Dict[str, str]:
        status = "available" if self.exists else "downloadable" if self.repo_id else "missing"
        return {
            "name": self.name,
            "path": self.path,
            "status": status,
            "description": self.description,
        }


@dataclass
class ControlNetRecord:
    name: str
    repo_id: str
    local_dir: str
    conditioning: str = "generic"
    subfolder: Optional[str] = None

    @property
    def exists(self) -> bool:
        return os.path.isdir(self.local_dir)

    @property
    def source(self) -> str:
        return self.local_dir if self.exists else self.repo_id


@dataclass
class IPAdapterRecord:
    name: str
    repo_id: str
    local_dir: str
    subfolder: Optional[str] = None
    weight_name: Optional[str] = None

    @property
    def exists(self) -> bool:
        return os.path.isdir(self.local_dir)

    @property
    def source(self) -> str:
        return self.local_dir if self.exists else self.repo_id

def _default_catalog() -> Dict[str, List[dict]]:
    return {
        "loras": [
            {
                "name": "SDXL Offset Example",
                "description": "Official SDXL offset LoRA for sharper detail (Stability AI).",
                "repo_id": "stabilityai/sd_xl_offset_example-lora_1.0",
                "filename": "sd_xl_offset_example-lora_1.0.safetensors",
                "path": "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_offset_example-lora_1.0.safetensors",
                "preview_url": "https://huggingface.co/stabilityai/sd_xl_offset_example-lora_1.0/resolve/main/An_example_image.png",
                "tags": ["sdxl", "detail", "general"],
            },
            {
                "name": "Pixel Art Character XL",
                "description": "Clean pixel character rendering tuned for SDXL.",
                "repo_id": "CrucibleAI/Pixel-Art-XL-LoRA",
                "filename": "pixel-art-xl.safetensors",
                "path": "lora/pixel-art-xl.safetensors",
                "preview_url": "https://huggingface.co/CrucibleAI/Pixel-Art-XL-LoRA/resolve/main/pixel_art_xl_sample.png",
                "tags": ["sdxl", "pixel art", "character"],
            },
            {
                "name": "Anime Companion SDXL",
                "description": "Stylised anime companion character LoRA for SDXL.",
                "repo_id": "Meina/AnimeCharacterSDXL-LoRA",
                "filename": "anime-companion.safetensors",
                "path": "lora/anime-companion.safetensors",
                "preview_url": "https://huggingface.co/Meina/AnimeCharacterSDXL-LoRA/resolve/main/preview.png",
                "tags": ["sdxl", "anime", "character"],
            },
        ]
    }


def _load_catalog_blob() -> Dict[str, List[dict]]:
    if os.path.isfile(CATALOG_FILE):
        with open(CATALOG_FILE, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    if "loras" not in data:
        data = _default_catalog()
    return data


def _default_controlnet_catalog() -> Dict[str, List[dict]]:
    return {
        "controlnets": [
            {
                "name": "Canny SDXL",
                "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
                "path": "canny",
                "conditioning": "canny",
            },
            {
                "name": "Depth SDXL",
                "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
                "path": "depth",
                "conditioning": "depth",
            },
        ]
    }


def _default_ip_adapter_catalog() -> Dict[str, List[dict]]:
    return {
        "ip_adapters": [
            {
                "name": "IP-Adapter SDXL",
                "repo_id": "TencentARC/IP-Adapter",
                "path": "ip_adapter_sdxl",
                "subfolder": "sdxl_models",
                "weight_name": "ip-adapter_sdxl.safetensors",
            }
        ]
    }


def _load_controlnet_catalog() -> Dict[str, List[dict]]:
    if os.path.isfile(CONTROLNET_CATALOG_FILE):
        with open(CONTROLNET_CATALOG_FILE, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    if "controlnets" not in data:
        data = _default_controlnet_catalog()
    return data


def _load_ip_adapter_catalog() -> Dict[str, List[dict]]:
    if os.path.isfile(IP_ADAPTER_CATALOG_FILE):
        with open(IP_ADAPTER_CATALOG_FILE, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    if "ip_adapters" not in data:
        data = _default_ip_adapter_catalog()
    return data


def _scan_local_files() -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for root, _dirs, files in os.walk(MODELS):
        for fname in files:
            if not fname.lower().endswith(SUPPORTED_EXTS):
                continue
            full = os.path.join(root, fname)
            found.setdefault(fname, []).append(full)
    return found


def list_records() -> List[LoraRecord]:
    blob = _load_catalog_blob()
    local_files = _scan_local_files()
    records: List[LoraRecord] = []
    seen_paths: set[str] = set()

    for entry in blob.get("loras", []):
        filename = entry.get("filename") or entry.get("file")
        if not filename:
            continue
        rel_path = entry.get("path") or os.path.join("lora", filename)
        normalized = rel_path.replace("\\", "/").lstrip("./")
        abs_path = os.path.join(MODELS, normalized)
        exists = os.path.exists(abs_path)
        if not exists and filename in local_files:
            abs_path = local_files[filename][0]
            normalized = os.path.relpath(abs_path, MODELS).replace("\\", "/")
            exists = True
        record = LoraRecord(
            name=entry.get("name", filename),
            filename=filename,
            path=normalized,
            local_path=abs_path,
            exists=exists,
            description=entry.get("description", ""),
            repo_id=entry.get("repo_id"),
            preview_url=entry.get("preview_url"),
            tags=entry.get("tags"),
        )
        records.append(record)
        seen_paths.add(os.path.normpath(abs_path))

    for filename, paths in local_files.items():
        for abs_path in paths:
            if os.path.normpath(abs_path) in seen_paths:
                continue
            rel = os.path.relpath(abs_path, MODELS).replace("\\", "/")
            record = LoraRecord(
                name=filename,
                filename=filename,
                path=rel,
                local_path=abs_path,
                exists=True,
                description="Local LoRA",
            )
            records.append(record)
            seen_paths.add(os.path.normpath(abs_path))

    return records


def records_by_name() -> Dict[str, LoraRecord]:
    return {record.name: record for record in list_records()}


def find_record(name: str) -> Optional[LoraRecord]:
    return records_by_name().get(name)


def list_controlnets() -> List[ControlNetRecord]:
    blob = _load_controlnet_catalog()
    records: List[ControlNetRecord] = []
    for entry in blob.get("controlnets", []):
        repo_id = entry.get("repo_id")
        if not repo_id:
            continue
        path_suffix = entry.get("path") or repo_id.split("/")[-1]
        local_dir = os.path.join(CONTROLNET_DIR, path_suffix)
        records.append(
            ControlNetRecord(
                name=entry.get("name", repo_id),
                repo_id=repo_id,
                local_dir=local_dir,
                conditioning=entry.get("conditioning", "generic"),
                subfolder=entry.get("subfolder"),
            )
        )

    # include additional local directories
    if os.path.isdir(CONTROLNET_DIR):
        for item in os.listdir(CONTROLNET_DIR):
            path = os.path.join(CONTROLNET_DIR, item)
            if os.path.isdir(path) and not any(record.local_dir == path for record in records):
                records.append(
                    ControlNetRecord(
                        name=item,
                        repo_id=item,
                        local_dir=path,
                    )
                )
    return records


def controlnet_records_by_name() -> Dict[str, ControlNetRecord]:
    return {record.name: record for record in list_controlnets()}


def find_controlnet(name: str) -> Optional[ControlNetRecord]:
    return controlnet_records_by_name().get(name)


def list_ip_adapters() -> List[IPAdapterRecord]:
    blob = _load_ip_adapter_catalog()
    records: List[IPAdapterRecord] = []
    for entry in blob.get("ip_adapters", []):
        repo_id = entry.get("repo_id")
        if not repo_id:
            continue
        path_suffix = entry.get("path") or repo_id.split("/")[-1]
        local_dir = os.path.join(IP_ADAPTER_DIR, path_suffix)
        records.append(
            IPAdapterRecord(
                name=entry.get("name", repo_id),
                repo_id=repo_id,
                local_dir=local_dir,
                subfolder=entry.get("subfolder"),
                weight_name=entry.get("weight_name"),
            )
        )

    if os.path.isdir(IP_ADAPTER_DIR):
        for item in os.listdir(IP_ADAPTER_DIR):
            path = os.path.join(IP_ADAPTER_DIR, item)
            if os.path.isdir(path) and not any(record.local_dir == path for record in records):
                records.append(
                    IPAdapterRecord(
                        name=item,
                        repo_id=item,
                        local_dir=path,
                    )
                )
    return records


def ip_adapter_records_by_name() -> Dict[str, IPAdapterRecord]:
    return {record.name: record for record in list_ip_adapters()}


def find_ip_adapter(name: str) -> Optional[IPAdapterRecord]:
    return ip_adapter_records_by_name().get(name)


def find_by_filename(filename: str) -> Optional[LoraRecord]:
    filename = os.path.basename(filename)
    for record in list_records():
        if record.filename == filename:
            return record
    return None


def resolve_path(raw_path: str) -> Optional[LoraRecord]:
    if not raw_path:
        return None
    normalized = raw_path.replace("\\", "/")
    records = list_records()

    if os.path.isabs(raw_path):
        abs_path = os.path.normpath(raw_path)
        for record in records:
            if os.path.normpath(record.local_path) == abs_path:
                return record
        if os.path.exists(abs_path):
            rel = os.path.relpath(abs_path, MODELS).replace("\\", "/")
            if rel.startswith(".."):
                rel = abs_path
            return LoraRecord(
                name=os.path.basename(abs_path),
                filename=os.path.basename(abs_path),
                path=rel,
                local_path=abs_path,
                exists=True,
            )
        return None

    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("models/"):
        normalized = normalized[len("models/"):]
    candidates = [normalized, os.path.join("lora", os.path.basename(normalized))]
    for record in records:
        rec_path = record.path.replace("\\", "/")
        if rec_path == normalized or rec_path == candidates[0] or rec_path == candidates[1]:
            return record
        if os.path.basename(rec_path) == os.path.basename(normalized):
            return record
    abs_candidates = [
        os.path.join(MODELS, normalized),
        os.path.join(ROOT, normalized),
        os.path.join(MODELS, os.path.basename(normalized)),
    ]
    for abs_path in abs_candidates:
        if os.path.exists(abs_path):
            rel = os.path.relpath(abs_path, MODELS).replace("\\", "/")
            return LoraRecord(
                name=os.path.basename(abs_path),
                filename=os.path.basename(abs_path),
                path=rel,
                local_path=abs_path,
                exists=True,
            )
    return None


def download(name: str) -> str:
    record = find_record(name)
    if record is None:
        return f"LoRA '{name}' is not in the catalog."
    if record.exists and os.path.exists(record.local_path):
        return f"LoRA '{name}' already exists at {record.local_path}."
    if not record.repo_id or hf_hub_download is None:
        return f"LoRA '{name}' is not downloadable automatically."

    target_dir = os.path.dirname(record.local_path) or LORAS
    os.makedirs(target_dir, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=record.repo_id,
            filename=record.filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # pragma: no cover - network/runtime failures
        return f"Failed to download '{name}': {exc}"

    if path != record.local_path:
        os.makedirs(os.path.dirname(record.local_path), exist_ok=True)
        if os.path.abspath(path) != os.path.abspath(record.local_path):
            try:
                if os.path.exists(record.local_path):
                    os.remove(record.local_path)
                os.replace(path, record.local_path)
            except OSError:
                pass
    return f"Downloaded '{name}' to {record.local_path}."


def to_display_table(records: Optional[Iterable[LoraRecord]] = None) -> List[Dict[str, str]]:
    records = list(records) if records is not None else list_records()
    return [record.to_display() for record in records]
