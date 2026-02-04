import os
import json
import shutil
import torch
import contextlib
import io
import logging
import hashlib
import math
from datetime import datetime, timezone
import soundfile as sf
import numpy as np
import folder_paths
import comfy.model_management as mm
from server import PromptServer
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from .dataset import TTSDataset
from accelerate import Accelerator
from torch.optim import AdamW
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup
from transformers.utils import cached_file
from safetensors.torch import save_file, load_file

# Register Qwen3-TTS models folder with ComfyUI
QWEN3_TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS")
os.makedirs(QWEN3_TTS_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS", QWEN3_TTS_MODELS_DIR)

# Register Qwen3-TTS prompts folder for voice embeddings
QWEN3_TTS_PROMPTS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS", "prompts")
os.makedirs(QWEN3_TTS_PROMPTS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS-Prompts", QWEN3_TTS_PROMPTS_DIR)

# Model repo mappings
QWEN3_TTS_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "Qwen3-TTS-12Hz-0.6B-Base",
}

# Tokenizer repo mapping
QWEN3_TTS_TOKENIZERS = {
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": "Qwen3-TTS-Tokenizer-12Hz",
}

def get_local_model_path(repo_id: str) -> str:
    """Get the local path for a model/tokenizer in ComfyUI's models folder."""
    folder_name = QWEN3_TTS_MODELS.get(repo_id) or QWEN3_TTS_TOKENIZERS.get(repo_id) or repo_id.replace("/", "_")
    return os.path.join(QWEN3_TTS_MODELS_DIR, folder_name)

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"

def count_jsonl_lines(file_path: str) -> int:
    """Count lines in a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def load_cache_metadata(meta_path: str) -> dict | None:
    """Load cache metadata, return None if invalid."""
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata.get('version') != 1:
            return None
        return metadata
    except (json.JSONDecodeError, IOError):
        return None

def save_cache_metadata(meta_path: str, metadata: dict) -> None:
    """Save cache metadata to file."""
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def migrate_cached_model(repo_id: str, target_path: str) -> bool:
    """Check for model in HuggingFace/ModelScope cache and migrate to ComfyUI folder."""
    if os.path.exists(target_path) and os.listdir(target_path):
        return True  # Already exists in target
    
    # Check HuggingFace cache
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hf_model_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    if os.path.exists(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                source = os.path.join(snapshots_dir, snapshots[0])
                print(f"Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True
    
    # Check ModelScope cache
    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True
    
    return False

def download_model_to_comfyui(repo_id: str, source: str) -> str:
    """Download a model directly to ComfyUI's models folder."""
    target_path = get_local_model_path(repo_id)
    
    # First check if we can migrate from cache
    if migrate_cached_model(repo_id, target_path):
        print(f"Model available at: {target_path}")
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    
    if source == "ModelScope":
        from modelscope import snapshot_download
        print(f"Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} from HuggingFace to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    
    return target_path

def get_available_models() -> list:
    """Get list of available models (downloaded + all options)."""
    available = []
    for repo_id, folder_name in QWEN3_TTS_MODELS.items():
        local_path = os.path.join(QWEN3_TTS_MODELS_DIR, folder_name)
        if os.path.exists(local_path) and os.listdir(local_path):
            available.append(f"✓ {repo_id}")
        else:
            available.append(repo_id)
    return available

# Helper to convert audio to ComfyUI format
def convert_audio(wav, sr):
    # wav is (channels, samples) or just (samples)
    # ComfyUI audio format: {"waveform": tensor(1, channels, samples), "sample_rate": int}
    # But usually audio nodes expect (batch, samples, channels) or (batch, channels, samples)?
    # Standard LoadAudio in ComfyUI returns:
    # "audio": {"waveform": audio_tensor, "sample_rate": sample_rate}
    # audio_tensor is [batch, channels, samples] (usually batch=1)
    
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) # (1, samples) (channels=1)
    
    # Qwen outputs numpy float32 usually.
    # Check if stereo/mono. Qwen3-TTS is mono usually?
    # Ensure shape is [1, channels, samples] for ComfyUI
    if wav.shape[0] > wav.shape[1]: 
        # assume (samples, channels) - verify this assumption
        wav = wav.transpose(0, 1)
        
    # If it's just (samples,), we made it (1, samples). 
    # ComfyUI often expects [Batch, Channels, Samples]. 
    # Let's wrap in batch.
    wav = wav.unsqueeze(0) # (1, channels, samples)
    
    return {"waveform": wav, "sample_rate": sr}

def load_audio_input(audio_input):
    # audio_input is {"waveform": tensor, "sample_rate": int}
    # waveform can be various formats: [B, C, T], [C, T], [T]
    # We need (samples,) or (channels, samples) numpy for Qwen
    # Qwen accepts numpy array.

    if audio_input is None:
        return None

    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]

    print(f"load_audio_input received: waveform.shape={waveform.shape}, waveform.dim={waveform.dim()}")

    # Handle different waveform formats
    if waveform.dim() == 0:
        # 0-dimensional tensor - this causes "Len() of unsized object"
        raise ValueError(f"Waveform is 0-dimensional (scalar tensor). This causes 'Len() of unsized object' error.")

    elif waveform.dim() == 1:
        # Already [samples], just return as-is
        wav = waveform
        print(f"Format [samples], returning as-is")

    elif waveform.dim() == 2:
        # Could be [C, T] or [T, C]
        if waveform.shape[0] < waveform.shape[1]:
            # Likely [C, T] - channels first
            if waveform.shape[0] > 1:
                wav = torch.mean(waveform, dim=0)  # Mix to mono
            else:
                wav = waveform[0]  # Take single channel
            print(f"Format [C, T], mixed to mono: {wav.shape}")
        else:
            # Likely [T, C] - samples first (shouldn't happen normally)
            if waveform.shape[1] > 1:
                wav = torch.mean(waveform, dim=1)  # Mix to mono
            else:
                wav = waveform[:, 0]
            wav = wav.unsqueeze(0)  # Add channel dimension back
            print(f"Format [T, C], converted: {wav.shape}")

    elif waveform.dim() == 3:
        # Expected [B, C, T] - take first batch
        wav = waveform[0]  # (channels, samples)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0)  # Mix to mono
        else:
            wav = wav.squeeze(0)  # (samples,)
        print(f"Format [B, C, T], took first batch and mixed: {wav.shape}")

    else:
        raise ValueError(f"Unexpected waveform dimension: {waveform.dim()} (shape: {waveform.shape})")

    # Ensure we have a 1D tensor
    if wav.dim() > 1:
        wav = wav.squeeze()
        print(f"Squeezed to 1D: {wav.shape}")

    return (wav.numpy(), sr)

class Qwen3Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (list(QWEN3_TTS_MODELS.keys()), {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to local model or checkpoint. If checkpoint (no speech_tokenizer/), base model loads from repo_id first."}),
            }
        }

    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-TTS"

    def load_model(self, repo_id, source, precision, attention, local_model_path=""):
        device = mm.get_torch_device()

        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        checkpoint_path = None
        local_path_stripped = local_model_path.strip() if local_model_path else ""

        if local_path_stripped:
            speech_tokenizer_path = os.path.join(local_path_stripped, "speech_tokenizer")
            if os.path.exists(speech_tokenizer_path):
                model_path = local_path_stripped
                print(f"Loading full model from: {model_path}")
            else:
                checkpoint_path = local_path_stripped
                local_path = get_local_model_path(repo_id)
                if os.path.exists(local_path) and os.listdir(local_path):
                    model_path = local_path
                else:
                    print(f"Base model not found locally. Downloading {repo_id}...")
                    model_path = download_model_to_comfyui(repo_id, source)
                print(f"Loading base model from: {model_path}")
                print(f"Will apply checkpoint from: {checkpoint_path}")
        else:
            local_path = get_local_model_path(repo_id)
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"Loading from ComfyUI models folder: {model_path}")
            else:
                print(f"Model not found locally. Downloading {repo_id}...")
                model_path = download_model_to_comfyui(repo_id, source)

        print(f"Loading Qwen3-TTS model on {device} as {dtype}")

        attn_impl = "sdpa"
        if attention != "auto":
            attn_impl = attention
        else:
            try:
                import flash_attn
                import importlib.metadata
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"

        print(f"Using attention implementation: {attn_impl}")

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )

        if checkpoint_path:
            ckpt_weights = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(ckpt_weights):
                state_dict = torch.load(ckpt_weights, map_location="cpu")
                model.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded checkpoint weights from {ckpt_weights}")
            else:
                raise ValueError(f"Checkpoint weights not found: {ckpt_weights}")

        # FORCE SPEAKER MAPPING FIX - Deep Injection
        try:
            cfg_file = os.path.join(checkpoint_path, "config.json") if checkpoint_path else os.path.join(model_path, "config.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
                
                if "talker_config" in cfg_data and "spk_id" in cfg_data["talker_config"]:
                    new_spk_id = cfg_data["talker_config"]["spk_id"]
                    new_spk_dialect = cfg_data["talker_config"].get("spk_is_dialect", {})
                    
                    # Target List: where spk_id might be hidden
                    configs_to_update = []
                    
                    # 1. Main model wrapper config
                    if hasattr(model, "config"): configs_to_update.append(model.config)
                    # 2. Internal model config
                    if hasattr(model, "model") and hasattr(model.model, "config"): configs_to_update.append(model.model.config)
                    
                    found_any = False
                    for root_cfg in configs_to_update:
                        # Try to find talker_config within these
                        t_cfg = getattr(root_cfg, "talker_config", None)
                        if t_cfg is not None:
                            for attr, val in [("spk_id", new_spk_id), ("spk_is_dialect", new_spk_dialect)]:
                                if not hasattr(t_cfg, attr) or getattr(t_cfg, attr) is None:
                                    setattr(t_cfg, attr, {})
                                cur_val = getattr(t_cfg, attr)
                                if isinstance(cur_val, dict):
                                    cur_val.update(val)
                                    found_any = True
                    
                    # 3. Direct access to the Talker's internal config (Most important)
                    if hasattr(model, "model") and hasattr(model.model, "talker") and hasattr(model.model.talker, "config"):
                        st_cfg = model.model.talker.config
                        for attr, val in [("spk_id", new_spk_id), ("spk_is_dialect", new_spk_dialect)]:
                            if not hasattr(st_cfg, attr) or getattr(st_cfg, attr) is None:
                                setattr(st_cfg, attr, {})
                            cur_val = getattr(st_cfg, attr)
                            if isinstance(cur_val, dict):
                                cur_val.update(val)
                                found_any = True
                    
                    if found_any:
                        print(f"DEBUG: Successfully injected custom speaker mapping: {new_spk_id}", flush=True)
                    else:
                        print("DEBUG: Failed to find an appropriate config object to inject mapping into.", flush=True)

                # Inject tts_model_type if present in checkpoint config
                if "tts_model_type" in cfg_data:
                    new_tts_model_type = cfg_data["tts_model_type"]

                    # Inject into config objects
                    for root_cfg in configs_to_update:
                        if hasattr(root_cfg, "tts_model_type"):
                            setattr(root_cfg, "tts_model_type", new_tts_model_type)

                    # CRITICAL: Also update the direct attribute on the inner model
                    # This is what generate_custom_voice() actually checks
                    if hasattr(model, "model") and hasattr(model.model, "tts_model_type"):
                        model.model.tts_model_type = new_tts_model_type

                    print(f"DEBUG: Injected tts_model_type = {new_tts_model_type}", flush=True)
        except Exception as e:
            print(f"DEBUG: Error during deep speaker injection: {e}", flush=True)
        
        return (model,)


class Qwen3CustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "speaker": ([
                    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", 
                    "Ryan", "Aiden", "Ono_Anna", "Sohee"
                ], {"default": "Vivian"}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                "custom_speaker_name": ("STRING", {"default": ""}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, language, speaker, seed, instruct="", custom_speaker_name="", max_new_tokens=8192):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, language, speaker, seed, instruct="", custom_speaker_name="", max_new_tokens=8192):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        lang = language if language != "Auto" else None
        inst = instruct if instruct.strip() != "" else None
        
        target_speaker = speaker
        if custom_speaker_name and custom_speaker_name.strip() != "":
            target_speaker = custom_speaker_name.strip()
            print(f"Using custom speaker: {target_speaker}")
        
        # Manual lookup and case-matching to bypass library validation failures
        try:
            configs_to_check = []
            if hasattr(model, "config"): configs_to_check.append(model.config)
            if hasattr(model, "model") and hasattr(model.model, "config"): configs_to_check.append(model.model.config)
            
            for root_cfg in configs_to_check:
                t_cfg = getattr(root_cfg, "talker_config", None)
                if t_cfg:
                    spk_map = getattr(t_cfg, "spk_id", None)
                    if isinstance(spk_map, dict):
                        # Case-insensitive match
                        match = next((s for s in spk_map if s.lower() == target_speaker.lower()), None)
                        if match:
                            print(f"DEBUG: Found case-matched speaker: '{match}' (original: '{target_speaker}')", flush=True)
                            target_speaker = match # Use the name the model expects
                            break
        except Exception as e:
            print(f"DEBUG: Speaker case-matching failed: {e}", flush=True)

        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=lang,
                speaker=target_speaker,
                instruct=inst,
                max_new_tokens=max_new_tokens
            )
        except ValueError as e:
            # Catch model type mismatch errors from qwen-tts
            msg = str(e)
            if "does not support generate_custom_voice" in msg:
                raise ValueError("Model Type Error: You are trying to use 'Custom Voice' with an incompatible model. Please load a 'CustomVoice' model (e.g. Qwen3-TTS-12Hz-1.7B-CustomVoice).") from e
            raise e
            
        return (convert_audio(wavs[0], sr),)


class Qwen3VoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "instruct": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, instruct, language, seed):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, instruct, language, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        lang = language if language != "Auto" else None
        
        try:
            wavs, sr = model.generate_voice_design(
                text=text,
                language=lang,
                instruct=instruct
            )
        except ValueError as e:
             msg = str(e)
             if "does not support generate_voice_design" in msg:
                 raise ValueError("Model Type Error: You are trying to use 'Voice Design' with an incompatible model. Please load a 'VoiceDesign' model (e.g. Qwen3-TTS-12Hz-1.7B-VoiceDesign).") from e
             raise e
             
        return (convert_audio(wavs[0], sr),)


class Qwen3PromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "ref_audio_max_seconds": ("FLOAT", {"default": 30.0, "min": -1.0, "max": 120.0, "step": 5.0}),
            }
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"

    def create_prompt(self, model, ref_audio, ref_text, ref_audio_max_seconds=30.0):
        audio_tuple = load_audio_input(ref_audio)
        
        # Trim reference audio if too long to prevent generation hangs (-1 = no limit)
        if audio_tuple is not None and ref_audio_max_seconds > 0:
            wav_data, audio_sr = audio_tuple
            max_samples = int(ref_audio_max_seconds * audio_sr)
            if len(wav_data) > max_samples:
                print(f"Trimming reference audio from {len(wav_data)/audio_sr:.1f}s to {ref_audio_max_seconds}s to prevent generation issues")
                wav_data = wav_data[:max_samples]
                audio_tuple = (wav_data, audio_sr)
        
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=audio_tuple,
                ref_text=ref_text
            )
        except ValueError as e:
             msg = str(e)
             # Assumption: create_voice_clone_prompt might also be restricted to Base models? 
             # README doesn't explicitly restrict it but implies it's for cloning.
             if "does not support" in msg:
                 raise ValueError("Model Type Error: This model does not support creating voice clone prompts. Please load a 'Base' model.") from e
             raise e
             
        return (prompt,)


class Qwen3SavePrompt:
    """Save a QWEN3_PROMPT (voice clone embedding) to disk as safetensors."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("QWEN3_PROMPT",),
                "filename": ("STRING", {"default": "voice_embedding"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_prompt"
    CATEGORY = "Qwen3-TTS"
    OUTPUT_NODE = True

    def save_prompt(self, prompt, filename):
        # prompt is List[VoiceClonePromptItem], we save the first item
        if not prompt or len(prompt) == 0:
            raise ValueError("Empty prompt - nothing to save")
        
        item = prompt[0]
        
        # Build tensors dict for safetensors
        tensors = {
            "ref_spk_embedding": item.ref_spk_embedding.contiguous().cpu(),
        }
        if item.ref_code is not None:
            tensors["ref_code"] = item.ref_code.contiguous().cpu()
        
        # Build metadata dict (safetensors metadata must be strings)
        metadata = {
            "x_vector_only_mode": str(item.x_vector_only_mode),
            "icl_mode": str(item.icl_mode),
        }
        if item.ref_text is not None:
            metadata["ref_text"] = item.ref_text
        
        # Ensure filename has no extension (we add .safetensors)
        if filename.endswith(".safetensors"):
            filename = filename[:-12]
        
        filepath = os.path.join(QWEN3_TTS_PROMPTS_DIR, f"{filename}.safetensors")
        
        save_file(tensors, filepath, metadata=metadata)
        print(f"Saved voice prompt to: {filepath}")
        
        return (filepath,)


class Qwen3LoadPrompt:
    """Load a QWEN3_PROMPT (voice clone embedding) from disk."""
    
    @classmethod
    def INPUT_TYPES(s):
        # Get list of available prompt files
        prompt_files = []
        if os.path.exists(QWEN3_TTS_PROMPTS_DIR):
            for f in os.listdir(QWEN3_TTS_PROMPTS_DIR):
                if f.endswith(".safetensors"):
                    prompt_files.append(f)
        if not prompt_files:
            prompt_files = ["no prompts saved yet"]
        
        return {
            "required": {
                "prompt_file": (sorted(prompt_files),),
            },
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Qwen3-TTS"

    @classmethod
    def IS_CHANGED(s, prompt_file):
        # Return file modification time to detect changes
        filepath = os.path.join(QWEN3_TTS_PROMPTS_DIR, prompt_file)
        if os.path.exists(filepath):
            return os.path.getmtime(filepath)
        return float("nan")

    def load_prompt(self, prompt_file):
        if prompt_file == "no prompts saved yet":
            raise ValueError("No prompt files available. Save a prompt first using Qwen3-TTS Save Prompt.")
        
        filepath = os.path.join(QWEN3_TTS_PROMPTS_DIR, prompt_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        
        # Load tensors
        tensors = load_file(filepath)
        
        # Load metadata
        from safetensors import safe_open
        with safe_open(filepath, framework="pt") as f:
            metadata = f.metadata() or {}
        
        # Reconstruct VoiceClonePromptItem
        ref_spk_embedding = tensors["ref_spk_embedding"]
        ref_code = tensors.get("ref_code", None)
        x_vector_only_mode = metadata.get("x_vector_only_mode", "False") == "True"
        icl_mode = metadata.get("icl_mode", "False") == "True"
        ref_text = metadata.get("ref_text", None)
        
        item = VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
            ref_text=ref_text,
        )
        
        print(f"Loaded voice prompt from: {filepath}")
        return ([item],)


class Qwen3VoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 42, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
                "prompt": ("QWEN3_PROMPT",),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "ref_audio_max_seconds": ("FLOAT", {"default": 30.0, "min": -1.0, "max": 120.0, "step": 5.0}),
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, seed, language="Auto", ref_audio=None, ref_text=None, prompt=None, max_new_tokens=2048, ref_audio_max_seconds=30.0):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, seed, language="Auto", ref_audio=None, ref_text=None, prompt=None, max_new_tokens=2048, ref_audio_max_seconds=30.0):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        lang = language if language != "Auto" else None
        
        wavs = None
        sr = 0
        
        try:
            if prompt is not None:
                # Use pre-calculated prompt
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    voice_clone_prompt=prompt,
                    max_new_tokens=max_new_tokens
                )
            elif ref_audio is not None and ref_text is not None and ref_text.strip() != "":
                # Use on-the-fly prompt creation
                audio_tuple = load_audio_input(ref_audio)
                
                # Trim reference audio if too long to prevent generation hangs (-1 = no limit)
                if audio_tuple is not None and ref_audio_max_seconds > 0:
                    wav_data, audio_sr = audio_tuple
                    max_samples = int(ref_audio_max_seconds * audio_sr)
                    if len(wav_data) > max_samples:
                        print(f"Trimming reference audio from {len(wav_data)/audio_sr:.1f}s to {ref_audio_max_seconds}s to prevent generation issues")
                        wav_data = wav_data[:max_samples]
                        audio_tuple = (wav_data, audio_sr)
                
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=audio_tuple,
                    ref_text=ref_text,
                    max_new_tokens=max_new_tokens
                )
            else:
                 raise ValueError("For Voice Clone, you must provide either 'prompt' OR ('ref_audio' AND 'ref_text').")
        except ValueError as e:
            msg = str(e)
            if "does not support generate_voice_clone" in msg:
                raise ValueError("Model Type Error: You are trying to use 'Voice Clone' with an incompatible model. Please load a 'Base' model (e.g. Qwen3-TTS-12Hz-1.7B-Base).") from e
            raise e
             
        return (convert_audio(wavs[0], sr),)

class Qwen3DatasetFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
                "output_filename": ("STRING", {"default": "dataset.jsonl", "multiline": False}),
                "ref_audio_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jsonl_path",)
    FUNCTION = "create_dataset"
    CATEGORY = "Qwen3-TTS/FineTuning"

    def create_dataset(self, folder_path, output_filename, ref_audio_path):
        folder_path = folder_path.strip().strip('"')
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
            
        jsonl_path = os.path.join(folder_path, output_filename)
        print(f"Creating dataset at: {jsonl_path}")
        
        # Get all files first to help matching
        all_files = os.listdir(folder_path)
        wav_files = [f for f in all_files if f.lower().endswith('.wav')]
        
        if not wav_files:
             raise ValueError(f"No .wav files found in {folder_path}")

        if not ref_audio_path or not os.path.exists(ref_audio_path):
            # Try to find default ref.wav
            possible_ref = os.path.join(folder_path, "ref.wav")
            if os.path.exists(possible_ref):
                ref_audio_path = possible_ref
            else:
                # Fallback to first wav?
                print("No ref.wav found and no ref_audio_path provided. Using the first wav file as reference (warning: this might include it in training context).")
                ref_audio_path = os.path.join(folder_path, wav_files[0])
        
        full_ref_path = os.path.abspath(ref_audio_path)
        print(f"Reference Audio: {full_ref_path}")
        
        count = 0
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for wav_file in wav_files:
                wav_path = os.path.join(folder_path, wav_file)
                
                # Check if this is the reference audio
                if os.path.abspath(wav_path) == full_ref_path:
                    continue
                    
                base_name = os.path.splitext(wav_file)[0]
                
                # Try finding text file with case matching or mismatch
                # We look for base_name + .txt (case insensitive) in the file list
                found_txt = None
                expected_txt_lower = (base_name + ".txt").lower()
                
                for cand in all_files:
                    if cand.lower() == expected_txt_lower:
                        found_txt = cand
                        break
                
                if not found_txt:
                    print(f"Skipping {wav_file}: Expected text file '{base_name}.txt' not found in {folder_path}")
                    # Debug: print what we have
                    # print(f"Available files: {all_files}") 
                    continue
                    
                txt_path = os.path.join(folder_path, found_txt)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
                    continue
                
                if not text:
                    print(f"Skipping {wav_file}: {found_txt} is empty.")
                    continue

                entry = {
                    "audio": os.path.abspath(wav_path),
                    "text": text,
                    "ref_audio": full_ref_path
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                
        if count == 0:
            print("Warning: No valid samples were added to the dataset!")
        else:
            print(f"Dataset created with {count} samples at {jsonl_path}")
            
        return (jsonl_path,)

class Qwen3DataPrep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "jsonl_path": ("STRING", {"default": "", "multiline": False}),
                "tokenizer_repo": (list(QWEN3_TTS_TOKENIZERS.keys()), {"default": "Qwen/Qwen3-TTS-Tokenizer-12Hz"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 32, "tooltip": "Number of audio files to process at once. Lower values use less VRAM."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_jsonl_path",)
    FUNCTION = "process"
    CATEGORY = "Qwen3-TTS/FineTuning"

    def process(self, jsonl_path, tokenizer_repo, source, batch_size, unique_id=None):
        # Helper to send progress text to UI
        def send_status(text):
            if unique_id:
                PromptServer.instance.send_progress_text(text, unique_id)

        device = mm.get_torch_device()

        output_path = jsonl_path.replace(".jsonl", "_codes.jsonl")
        meta_path = jsonl_path.replace(".jsonl", "_codes.meta.json")

        send_status("Checking cache...")
        input_hash = compute_file_hash(jsonl_path)
        input_line_count = count_jsonl_lines(jsonl_path)

        # Check cache validity
        if os.path.exists(output_path):
            metadata = load_cache_metadata(meta_path)
            if metadata:
                if (metadata.get('input_hash') == input_hash and
                    metadata.get('tokenizer_repo') == tokenizer_repo and
                    metadata.get('output_line_count') == input_line_count):
                    # Verify output file integrity
                    if count_jsonl_lines(output_path) == metadata.get('output_line_count'):
                        print(f"[Qwen3DataPrep] Cache hit - using existing processed data")
                        send_status("Using cached data (no reprocessing needed)")
                        return (output_path,)
                print(f"[Qwen3DataPrep] Cache invalid, reprocessing...")
            else:
                print(f"[Qwen3DataPrep] No valid cache metadata, reprocessing...")
        else:
            print(f"[Qwen3DataPrep] No cached output found, will process")

        # Resolve tokenizer path - check ComfyUI folder first, download if needed
        local_path = get_local_model_path(tokenizer_repo)
        if os.path.exists(local_path) and os.listdir(local_path):
            tokenizer_path = local_path
            print(f"Loading Tokenizer from ComfyUI folder: {tokenizer_path}")
        else:
            print(f"Tokenizer not found locally. Downloading {tokenizer_repo}...")
            tokenizer_path = download_model_to_comfyui(tokenizer_repo, source)

        send_status("Loading tokenizer...")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_path,
            device_map=device,
        )

        inputs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                inputs.append(json.loads(line.strip()))

        total_items = len(inputs)
        total_batches = (total_items + batch_size - 1) // batch_size
        print(f"Processing {total_items} items in {total_batches} batches (batch_size={batch_size})...")

        # Write results incrementally to avoid memory accumulation
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for batch_idx, i in enumerate(range(0, total_items, batch_size)):
                batch = inputs[i:i+batch_size]
                audio_paths = [b['audio'] for b in batch]

                status_msg = f"Processing batch {batch_idx + 1}/{total_batches}..."
                print(status_msg)
                send_status(status_msg)

                # Encode audio files
                enc_res = tokenizer.encode(audio_paths)
                codes = enc_res.audio_codes

                # Write batch results immediately to disk
                for j, code in enumerate(codes):
                    item = batch[j]
                    item['audio_codes'] = code.cpu().tolist()
                    out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

                # Clear VRAM between batches to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save cache metadata
        metadata = {
            "version": 1,
            "input_hash": input_hash,
            "tokenizer_repo": tokenizer_repo,
            "input_line_count": input_line_count,
            "output_line_count": len(inputs),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        save_cache_metadata(meta_path, metadata)

        print(f"Processed dataset saved to {output_path}")
        send_status("Data preparation complete!")
        return (output_path,)

class Qwen3FineTune:
    @classmethod
    def INPUT_TYPES(s):
        # Get base models (excluding CustomVoice/VoiceDesign for fine-tuning)
        base_models = [k for k in QWEN3_TTS_MODELS.keys() if "Base" in k]
        return {
            "required": {
                "train_jsonl": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to the preprocessed JSONL file containing training data with audio codes."}),
                "init_model": (base_models, {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "tooltip": "Base model to fine-tune. Must be a 'Base' model variant."}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace", "tooltip": "Download source if model is not found locally."}),
                "output_dir": ("STRING", {"default": "output/finetuned_model", "multiline": False, "tooltip": "Directory to save checkpoints and final model."}),
                "epochs": ("INT", {"default": 3, "min": 1, "max": 1000, "tooltip": "Number of training epochs to run."}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "Number of samples per batch. Lower values use less VRAM."}),
                "lr": ("FLOAT", {"default": 2e-6, "step": 1e-7, "tooltip": "Learning rate. Qwen default (2e-5) is too aggressive for small batches, causing noise output. Defaults to 2e-6 for stability."}),
                "speaker_name": ("STRING", {"default": "my_speaker", "tooltip": "Name for the custom speaker. Use this name when generating with the fine-tuned model."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility."}),
            },
            "optional": {
                 # Workflow
                 "resume_training": ("BOOLEAN", {"default": False, "tooltip": "Continue training from the latest checkpoint in output_dir."}),
                 "log_every_steps": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Log training progress every N steps."}),
                 "save_every_epochs": ("INT", {"default": 1, "min": 0, "max": 100, "tooltip": "Save checkpoint every N epochs. Set to 0 to only save final epoch. Ignored if save_every_steps > 0."}),
                 "save_every_steps": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Save checkpoint every N steps. Set to 0 to use epoch-based saving instead."}),
                 # VRAM Optimizations
                 "mixed_precision": (["bf16", "fp32"], {"default": "bf16", "tooltip": "bf16 recommended. Use fp32 only if GPU doesn't support bf16 (pre-Ampere)."}),
                 "gradient_accumulation": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Accumulate gradients over N steps before updating. Effective batch size = batch_size * gradient_accumulation."}),
                 "gradient_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "Trade compute for VRAM by recomputing activations. Saves ~30-40% VRAM."}),
                 "use_8bit_optimizer": ("BOOLEAN", {"default": True, "tooltip": "Use 8-bit AdamW optimizer. Saves ~50% optimizer VRAM. Requires bitsandbytes."}),
                 # Training Dynamics
                 "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "L2 regularization strength to prevent overfitting."}),
                 "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Gradient clipping threshold to prevent exploding gradients."}),
                 # Learning Rate Schedule
                 "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Number of warmup steps. Set to 0 to disable warmup. Recommended: 5-10% of total steps."}),
                 "warmup_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Warmup as ratio of total steps. Ignored if warmup_steps > 0. E.g., 0.1 = 10% warmup."}),
                 "save_optimizer_state": ("BOOLEAN", {"default": False, "tooltip": "Save optimizer/scheduler state in checkpoints. Enables perfect resume but doubles checkpoint size."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "custom_speaker_name")
    FUNCTION = "train"
    CATEGORY = "Qwen3-TTS/FineTuning"

    def train(self, train_jsonl, init_model, source, output_dir, epochs, batch_size, lr, speaker_name, seed, mixed_precision="bf16", resume_training=False, log_every_steps=10, save_every_epochs=1, save_every_steps=0, gradient_accumulation=4, gradient_checkpointing=True, use_8bit_optimizer=True, weight_decay=0.01, max_grad_norm=1.0, warmup_steps=0, warmup_ratio=0.0, save_optimizer_state=False, unique_id=None):
        # Helper to send progress text to UI
        def send_status(text):
            if unique_id:
                PromptServer.instance.send_progress_text(text, unique_id)

        # Setup output directory
        full_output_dir = os.path.abspath(output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        # Check for resume checkpoint
        start_epoch = 0
        resume_from_step = 0  # Track step offset for ckpt_step_N checkpoints
        resume_checkpoint_path = None

        if resume_training:
            # Priority 1: Find checkpoint subfolders (prefer trained-on checkpoints)
            checkpoints = []
            if os.path.exists(full_output_dir):
                for item in os.listdir(full_output_dir):
                    item_path = os.path.join(full_output_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
                        mtime = os.path.getmtime(os.path.join(item_path, "pytorch_model.bin"))
                        checkpoints.append((mtime, item, item_path))

            if checkpoints:
                # Sort by mtime (most recent first)
                checkpoints.sort(key=lambda x: x[0], reverse=True)
                _, item_name, resume_checkpoint_path = checkpoints[0]

                # Extract epoch OR step number from folder name
                if item_name.startswith("epoch_"):
                    try:
                        start_epoch = int(item_name.split("_")[1])
                    except (ValueError, IndexError):
                        pass
                elif item_name.startswith("ckpt_step_"):
                    try:
                        resume_from_step = int(item_name.split("_")[2])
                    except (ValueError, IndexError):
                        pass

                print(f"Resume: Found checkpoint '{item_name}' (most recent)")
            else:
                # Priority 2: Check if output_dir itself is a checkpoint
                direct_weights = os.path.join(full_output_dir, "pytorch_model.bin")
                if os.path.exists(direct_weights):
                    resume_checkpoint_path = full_output_dir
                    dir_name = os.path.basename(full_output_dir)
                    if dir_name.startswith("ckpt_step_"):
                        try:
                            resume_from_step = int(dir_name.split("_")[2])
                        except (ValueError, IndexError):
                            pass
                    print(f"Resume: output_dir is a checkpoint, loading from {resume_checkpoint_path}")

            # Load step_offset from checkpoint's training_config.json if not extracted from folder name
            if resume_checkpoint_path and resume_from_step == 0:
                training_config_path = os.path.join(resume_checkpoint_path, "training_config.json")
                if os.path.exists(training_config_path):
                    with open(training_config_path, 'r') as f:
                        saved_config = json.load(f)
                        resume_from_step = saved_config.get("step_offset", 0)
                        if resume_from_step > 0:
                            print(f"Loaded step_offset={resume_from_step} from checkpoint config")

            if resume_checkpoint_path:
                if resume_from_step > 0:
                    print(f"Will continue from step {resume_from_step}")
                print(f"Will train epochs {start_epoch + 1} to {start_epoch + epochs}")
            else:
                print("Resume enabled but no checkpoints found, starting fresh")

        # Resolve init_model path - check ComfyUI folder first, download if needed
        # NOTE: Always use the original base model, not checkpoint - checkpoint's model.safetensors
        # doesn't include speaker_encoder (it's stripped for inference). We load checkpoint weights separately.
        if init_model in QWEN3_TTS_MODELS:
            local_path = get_local_model_path(init_model)
            if os.path.exists(local_path) and os.listdir(local_path):
                init_model_path = local_path
                print(f"Using model from ComfyUI folder: {init_model_path}")
            else:
                print(f"Base model not found locally. Downloading {init_model}...")
                init_model_path = download_model_to_comfyui(init_model, source)
        else:
            # Assume it's a path
            init_model_path = init_model

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ComfyUI runs in inference_mode by default.
        # We must disable it and enable gradients properly for the entire scope, including model loading.
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                # Clear VRAM before loading to maximize available memory
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                use_cpu = mm.cpu_mode()
                # Force single-process training mode by clearing distributed env vars.
                # This prevents NCCL init on Windows and "RANK expected" errors from stale env vars.
                for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
                    os.environ.pop(key, None)

                # Check GPU bf16 support and fallback to fp32 if needed
                actual_mixed_precision = mixed_precision
                if not use_cpu and torch.cuda.is_available() and mixed_precision == "bf16":
                    device_cap = torch.cuda.get_device_capability()
                    gpu_name = torch.cuda.get_device_name()
                    # bf16 requires compute capability >= 8.0 (Ampere+)
                    if device_cap[0] < 8:
                        print(f"Warning: {gpu_name} (compute {device_cap[0]}.{device_cap[1]}) does not support bf16.")
                        print("Falling back to fp32. Note: FP32 uses ~2x more VRAM than bf16.")
                        actual_mixed_precision = "fp32"

                # Accelerator uses "no" for fp32
                accel_precision = "no" if actual_mixed_precision == "fp32" else actual_mixed_precision
                accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation, mixed_precision=accel_precision, cpu=use_cpu)

                if resume_checkpoint_path:
                    print(f"Loading base model: {init_model_path} (will apply checkpoint weights from {resume_checkpoint_path})")
                else:
                    print(f"Loading base model: {init_model_path}")
                
                attn_impl = "sdpa"
                try:
                     import flash_attn
                     import importlib.metadata
                     importlib.metadata.version("flash_attn")
                     attn_impl = "flash_attention_2"
                except Exception:
                     pass

                print(f"Using attention implementation: {attn_impl}")

                dtype = torch.bfloat16 if actual_mixed_precision == "bf16" else torch.float32

                qwen3tts = Qwen3TTSModel.from_pretrained(
                    init_model_path,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )

                # Load training weights (includes speaker_encoder) if resuming
                if resume_checkpoint_path:
                    ckpt_weights = os.path.join(resume_checkpoint_path, "pytorch_model.bin")
                    if os.path.exists(ckpt_weights):
                        state_dict = torch.load(ckpt_weights, map_location="cpu")
                        qwen3tts.model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded training weights from {ckpt_weights}")
                    else:
                        print(f"Warning: Training checkpoint not found at {ckpt_weights}, using model.safetensors weights")

                # FORCE GRADIENTS ON
                qwen3tts.model.train()
                for name, param in qwen3tts.model.named_parameters():
                    param.requires_grad = True

                # Enable gradient checkpointing to reduce VRAM usage (~30-40% savings)
                if gradient_checkpointing:
                    if hasattr(qwen3tts.model, 'gradient_checkpointing_enable'):
                        qwen3tts.model.gradient_checkpointing_enable()
                        print("Gradient checkpointing enabled for VRAM optimization")
                    elif hasattr(qwen3tts.model, 'talker') and hasattr(qwen3tts.model.talker, 'gradient_checkpointing_enable'):
                        qwen3tts.model.talker.gradient_checkpointing_enable()
                        print("Gradient checkpointing enabled on talker for VRAM optimization")
                else:
                    print("Gradient checkpointing disabled")

                config = AutoConfig.from_pretrained(init_model_path)
                
                # Load Data
                with open(train_jsonl, 'r', encoding='utf-8') as f:
                    train_lines = [json.loads(line) for line in f]
                    
                dataset = TTSDataset(train_lines, qwen3tts.processor, config)
                generator = torch.Generator()
                generator.manual_seed(seed)

                train_dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=dataset.collate_fn,
                    generator=generator,
                )
                
                # Use 8-bit Adam if available and enabled (saves ~50% optimizer memory)
                if HAS_BNB and use_8bit_optimizer:
                    optimizer = bnb.optim.AdamW8bit(qwen3tts.model.parameters(), lr=lr, weight_decay=weight_decay)
                    print("Using 8-bit AdamW optimizer for VRAM optimization")
                else:
                    optimizer = AdamW(qwen3tts.model.parameters(), lr=lr, weight_decay=weight_decay)
                    if not HAS_BNB:
                        print("Using standard AdamW (install bitsandbytes for lower VRAM usage)")
                    else:
                        print("Using standard AdamW (8-bit optimizer disabled)")

                # Calculate total training steps for THIS run (use ceil to avoid 0 for small datasets)
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation)
                total_training_steps = num_update_steps_per_epoch * epochs

                # Determine warmup steps (explicit steps take priority over ratio)
                actual_warmup_steps = warmup_steps
                if warmup_steps == 0 and warmup_ratio > 0:
                    actual_warmup_steps = int(total_training_steps * warmup_ratio)

                # Create scheduler if warmup is enabled
                scheduler = None
                if actual_warmup_steps > 0:
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=actual_warmup_steps,
                        num_training_steps=total_training_steps
                    )
                    print(f"Using linear warmup scheduler: {actual_warmup_steps} warmup steps out of {total_training_steps} total")

                # Handle resume: restore optimizer and scheduler state if available
                if resume_checkpoint_path:
                    # Load optimizer state (important for momentum/Adam statistics)
                    optimizer_state_path = os.path.join(resume_checkpoint_path, "optimizer.pt")
                    if os.path.exists(optimizer_state_path):
                        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
                        print(f"Loaded optimizer state from {optimizer_state_path}")
                    else:
                        print("No optimizer state found, starting fresh (momentum will be reset)")

                    # Load scheduler state if using warmup
                    if scheduler:
                        scheduler_state_path = os.path.join(resume_checkpoint_path, "scheduler.pt")
                        if os.path.exists(scheduler_state_path):
                            scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu"))
                            print(f"Loaded scheduler state from {scheduler_state_path}")
                        else:
                            # Fast-forward scheduler to current position (for checkpoints saved before this feature)
                            completed_steps = start_epoch * num_update_steps_per_epoch
                            if completed_steps > 0:
                                print(f"Fast-forwarding scheduler by {completed_steps} steps (no saved state found)")
                                for _ in range(completed_steps):
                                    scheduler.step()

                if scheduler:
                    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
                        qwen3tts.model, optimizer, train_dataloader, scheduler
                    )
                else:
                    model, optimizer, train_dataloader = accelerator.prepare(
                        qwen3tts.model, optimizer, train_dataloader
                    )
                
                model.train()

                target_speaker_embedding = None

                # Calculate total epochs for this run
                end_epoch = start_epoch + epochs
                print(f"Starting training from epoch {start_epoch + 1} to {end_epoch}...")

                # Helper function to save a training checkpoint (also inference-ready)
                def save_training_checkpoint(checkpoint_name):
                    """Save checkpoint for resuming training. Also inference-ready."""
                    ckpt_path = os.path.join(full_output_dir, checkpoint_name)
                    os.makedirs(ckpt_path, exist_ok=True)

                    # Save training weights with speaker embedding injected
                    unwrapped = accelerator.unwrap_model(model)
                    state_dict = {k: v.cpu() for k, v in unwrapped.state_dict().items()}

                    # Inject speaker embedding at index 3000 (for inference)
                    if target_speaker_embedding is not None:
                        weight = state_dict['talker.model.codec_embedding.weight']
                        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().cpu().to(weight.dtype)

                    torch.save(state_dict, os.path.join(ckpt_path, "pytorch_model.bin"))

                    # Save config for inference (speaker mapping)
                    base_cfg_path = os.path.join(init_model_path, "config.json")
                    with open(base_cfg_path, 'r', encoding='utf-8') as f:
                        ckpt_cfg = json.load(f)

                    ckpt_cfg["tts_model_type"] = "custom_voice"
                    spk_key = speaker_name.lower()
                    ckpt_cfg["talker_config"]["spk_id"] = {spk_key: 3000}
                    ckpt_cfg["talker_config"]["spk_is_dialect"] = {spk_key: False}

                    with open(os.path.join(ckpt_path, "config.json"), 'w', encoding='utf-8') as f:
                        json.dump(ckpt_cfg, f, indent=2, ensure_ascii=False)

                    if save_optimizer_state:
                        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
                        if scheduler:
                            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

                    # Save training config with step_offset for resume
                    training_config = {
                        "step_offset": resume_from_step,
                    }
                    with open(os.path.join(ckpt_path, "training_config.json"), 'w') as f:
                        json.dump(training_config, f, indent=2)

                    print(f"Training checkpoint saved: {ckpt_path}")
                    return ckpt_path

                # Helper function to save final inference-ready model
                def save_final_model(checkpoint_name):
                    """Save complete model ready for inference and resume."""
                    ckpt_path = os.path.join(full_output_dir, checkpoint_name)

                    # Copy config files only (exclude speech_tokenizer, model files)
                    def ignore_files(directory, files):
                        ignored = set()
                        if directory == init_model_path:
                            if "speech_tokenizer" in files:
                                ignored.add("speech_tokenizer")
                            if "model.safetensors" in files:
                                ignored.add("model.safetensors")
                            if "pytorch_model.bin" in files:
                                ignored.add("pytorch_model.bin")
                        return ignored

                    shutil.copytree(init_model_path, ckpt_path, ignore=ignore_files, dirs_exist_ok=True)

                    # Modify config.json for custom voice
                    ckpt_cfg_path = os.path.join(ckpt_path, "config.json")
                    with open(ckpt_cfg_path, 'r', encoding='utf-8') as f:
                        ckpt_cfg = json.load(f)

                    ckpt_cfg["tts_model_type"] = "custom_voice"
                    spk_key = speaker_name.lower()
                    ckpt_cfg["talker_config"]["spk_id"] = {spk_key: 3000}
                    ckpt_cfg["talker_config"]["spk_is_dialect"] = {spk_key: False}

                    with open(ckpt_cfg_path, 'w', encoding='utf-8') as f:
                        json.dump(ckpt_cfg, f, indent=2, ensure_ascii=False)

                    # Save weights with speaker embedding injected (keeps speaker_encoder for resume)
                    unwrapped = accelerator.unwrap_model(model)
                    state_dict = {k: v.cpu() for k, v in unwrapped.state_dict().items()}

                    # Inject speaker embedding at index 3000 (for inference)
                    if target_speaker_embedding is not None:
                        weight = state_dict['talker.model.codec_embedding.weight']
                        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().cpu().to(weight.dtype)

                    # Save as pytorch_model.bin (works for both inference and resume)
                    torch.save(state_dict, os.path.join(ckpt_path, "pytorch_model.bin"))

                    if save_optimizer_state:
                        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
                        if scheduler:
                            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

                    # Save training config with step_offset for resume
                    training_config = {
                        "step_offset": resume_from_step,
                    }
                    with open(os.path.join(ckpt_path, "training_config.json"), 'w') as f:
                        json.dump(training_config, f, indent=2)

                    print(f"Final model saved: {ckpt_path}")
                    return ckpt_path

                # Calculate total optimizer steps and global step counter
                # Use num_update_steps_per_epoch (optimizer steps) not len(train_dataloader) (micro-batches)
                total_optimizer_steps = num_update_steps_per_epoch * end_epoch + resume_from_step
                global_step = start_epoch * num_update_steps_per_epoch + resume_from_step  # Resume from correct optimizer step

                for epoch in range(start_epoch, end_epoch):
                    epoch_loss = 0
                    steps = 0
                    send_status(f"Epoch {epoch + 1}/{end_epoch} - Training...")
                    for batch in train_dataloader:
                        with accelerator.accumulate(model):
                            # Debug info (only on first batch of first epoch in this run)
                            if steps == 0 and epoch == start_epoch:
                                 print(f"DEBUG: Grad Enabled: {torch.is_grad_enabled()}")
                                 print(f"DEBUG: Inference Mode: {torch.is_inference_mode_enabled()}")
                                 for n, p in model.named_parameters():
                                     if p.requires_grad:
                                         print(f"DEBUG: Parameter {n} requires grad.")
                                         break

                            # Data extraction logic from sft_12hz.py
                            input_ids = batch['input_ids']
                            codec_ids = batch['codec_ids']
                            ref_mels = batch['ref_mels']
                            text_embedding_mask = batch['text_embedding_mask']
                            codec_embedding_mask = batch['codec_embedding_mask']
                            attention_mask = batch['attention_mask']
                            codec_0_labels = batch['codec_0_labels']
                            codec_mask = batch['codec_mask']
                            
                            # Unwrap model to access attributes (DDP/FSDP wrappers hide them)
                            unwrapped_model = accelerator.unwrap_model(model)

                            # Get device/dtype from model parameters (DDP wrappers don't expose these directly)
                            model_dtype = next(unwrapped_model.parameters()).dtype
                            model_device = next(unwrapped_model.parameters()).device
                            speaker_embedding = unwrapped_model.speaker_encoder(ref_mels.to(model_device).to(model_dtype)).detach()
                            if target_speaker_embedding is None:
                                target_speaker_embedding = speaker_embedding

                            input_text_ids = input_ids[:, :, 0]
                            input_codec_ids = input_ids[:, :, 1]

                            # Use unwrapped model for attribute access (DDP/FSDP wrappers hide them)
                            current_model = unwrapped_model
                            
                            # Debug Gradient Flow
                            if steps == 0 and epoch == start_epoch:
                                print(f"DEBUG: Model Training Mode: {current_model.training}")
                                # Check embedding layer grad
                                emb_layer = current_model.talker.model.text_embedding
                                print(f"DEBUG: Text Embedding Layer Weight requires_grad: {emb_layer.weight.requires_grad}")

                            # 0.6B model requires text_projection for dimension matching (1024 -> 2048)
                            raw_text_embedding = current_model.talker.model.text_embedding(input_text_ids)
                            if "0.6B" in init_model:
                                input_text_embedding = current_model.talker.text_projection(raw_text_embedding) * text_embedding_mask
                            else:
                                input_text_embedding = raw_text_embedding * text_embedding_mask
                            input_codec_embedding = current_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                            input_codec_embedding[:, 6, :] = speaker_embedding
                            
                            input_embeddings = input_text_embedding + input_codec_embedding
                            
                            if steps == 0 and epoch == start_epoch:
                                 print(f"DEBUG: input_text_embedding requires_grad: {input_text_embedding.requires_grad}")
                                 print(f"DEBUG: input_codec_embedding requires_grad: {input_codec_embedding.requires_grad}")
                                 print(f"DEBUG: input_embeddings requires_grad: {input_embeddings.requires_grad}")
                            
                            for i in range(1, 16):
                                codec_i_embedding = current_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                                input_embeddings = input_embeddings + codec_i_embedding
                                
                            outputs = current_model.talker(
                                inputs_embeds=input_embeddings[:, :-1, :],
                                attention_mask=attention_mask[:, :-1],
                                labels=codec_0_labels[:, 1:],
                                output_hidden_states=True
                            )
                            
                            hidden_states = outputs.hidden_states[0][-1]
                            talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                            talker_codec_ids = codec_ids[codec_mask]
                            
                            sub_talker_logits, sub_talker_loss = current_model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                            
                            loss = outputs.loss + sub_talker_loss
                            
                            if steps == 0 and epoch == start_epoch:
                                print(f"DEBUG: Loss requires_grad: {loss.requires_grad}")
                                if not loss.requires_grad:
                                    print(f"DEBUG: outputs.loss requires_grad: {outputs.loss.requires_grad if outputs.loss is not None else 'None'}")
                                    print(f"DEBUG: sub_talker_loss requires_grad: {sub_talker_loss.requires_grad}")
                            
                            accelerator.backward(loss)

                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                            optimizer.step()
                            if scheduler:
                                scheduler.step()
                            optimizer.zero_grad()

                            epoch_loss += loss.item()
                            steps += 1

                            # Only count optimizer steps (after gradient accumulation completes)
                            if accelerator.sync_gradients:
                                global_step += 1

                                # Show step progress periodically
                                if log_every_steps > 0 and global_step % log_every_steps == 0:
                                    lr_val = optimizer.param_groups[0]['lr']
                                    status = f"Step {global_step}/{total_optimizer_steps}, Loss: {loss.item():.4f}, LR: {lr_val:.8f}"
                                    print(status)
                                    send_status(status)

                                # Step-based saving: only lightweight checkpoints during training
                                # (final model is always saved as epoch_N after training loop)
                                if save_every_steps > 0 and global_step % save_every_steps == 0:
                                    send_status(f"Saving checkpoint step {global_step}...")
                                    save_training_checkpoint(f"ckpt_step_{global_step}")

                    avg_loss = epoch_loss/steps if steps > 0 else 0
                    print(f"Epoch {epoch + 1}/{end_epoch} - Avg Loss: {avg_loss}")
                    send_status(f"Epoch {epoch + 1}/{end_epoch} - Loss: {avg_loss:.4f}")

                    # Epoch-based saving: intermediate checkpoints only (final saved after loop)
                    if save_every_steps == 0 and save_every_epochs > 0:
                        is_final_epoch = (epoch + 1) == end_epoch
                        should_save_checkpoint = ((epoch + 1) % save_every_epochs == 0) and not is_final_epoch
                        if should_save_checkpoint:
                            send_status(f"Saving checkpoint epoch {epoch + 1}...")
                            save_training_checkpoint(f"ckpt_epoch_{epoch + 1}")

                # Always save final model as epoch_N for consistent resume
                send_status(f"Saving final model epoch {end_epoch}...")
                save_final_model(f"epoch_{end_epoch}")
                final_output_path = os.path.join(full_output_dir, f"epoch_{end_epoch}")

                # Cleanup: free accelerator resources and synchronize CUDA
                accelerator.free_memory()
                del model, optimizer, train_dataloader, qwen3tts
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                print(f"Fine-tuning complete. Model saved to {final_output_path}")
                send_status("Training complete!")
                return (final_output_path, speaker_name)


class Qwen3AudioCompare:
    # Class-level cache for speaker encoder
    _speaker_encoder = None
    _speaker_encoder_cache_key = None

    @classmethod
    def INPUT_TYPES(s):
        # Get available Base models
        base_models = [k for k in QWEN3_TTS_MODELS.keys() if "Base" in k]
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "generated_audio": ("AUDIO",),
                "speaker_encoder_model": (base_models, {"default": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "tooltip": "Base model to load speaker encoder from (only loads ~76 weights, not the full model)"}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "", "tooltip": "Optional custom path to model directory. If empty, uses default models/Qwen3-TTS/ location."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "Qwen3-TTS/Evaluation"

    def _load_speaker_encoder(self, model_repo, local_model_path=""):
        """Load only the speaker encoder from a Base model (not the full model)."""
        from safetensors import safe_open
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

        # Get local model path - use provided path if non-empty, otherwise fall back to default
        if local_model_path and local_model_path.strip():
            model_path = os.path.abspath(local_model_path.strip())
        else:
            model_path = get_local_model_path(model_repo)

        # Check if already cached (use resolved path as cache key)
        if Qwen3AudioCompare._speaker_encoder is not None and Qwen3AudioCompare._speaker_encoder_cache_key == model_path:
            return Qwen3AudioCompare._speaker_encoder
        if not os.path.exists(model_path):
            raise ValueError(f"Base model not found at {model_path}. Please download it first using Qwen3-TTS Loader.")

        # Load config to get speaker encoder config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Validate this is a Base model with speaker encoder
        if "speaker_encoder_config" not in config_dict:
            raise ValueError(f"Model at {model_path} does not contain speaker_encoder_config. Only Base models (e.g., Qwen3-TTS-12Hz-0.6B-Base) include the speaker encoder.")

        # Create speaker encoder config
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        speaker_config = Qwen3TTSSpeakerEncoderConfig(**config_dict["speaker_encoder_config"])

        # Instantiate speaker encoder
        speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config)

        # Load only speaker encoder weights from safetensors (selective loading to save memory)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        speaker_weights = {}
        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("speaker_encoder."):
                    speaker_weights[key[len("speaker_encoder."):]] = f.get_tensor(key)

        speaker_encoder.load_state_dict(speaker_weights)
        speaker_encoder.eval()

        # Move to GPU if available
        device = mm.get_torch_device()
        speaker_encoder = speaker_encoder.to(device)

        # Cache it (use resolved path as key)
        Qwen3AudioCompare._speaker_encoder = speaker_encoder
        Qwen3AudioCompare._speaker_encoder_cache_key = model_path

        print(f"Loaded speaker encoder from {model_repo} ({len(speaker_weights)} weights)")
        return speaker_encoder

    def _extract_speaker_embedding(self, speaker_encoder, audio, sr):
        """Extract speaker embedding from audio using the speaker encoder."""
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        # Resample to 24kHz if needed
        if sr != 24000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=24000)

        # Compute mel spectrogram
        mel = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000
        ).transpose(1, 2)

        # Get embedding
        device = next(speaker_encoder.parameters()).device
        mel = mel.to(device)
        with torch.no_grad():
            embedding = speaker_encoder(mel)
        return embedding

    def compare(self, reference_audio, generated_audio, speaker_encoder_model, local_model_path=""):
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        # Extract waveforms from ComfyUI audio format
        def extract_wav(audio_input):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
            wav = waveform[0]  # Take first batch
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0)  # Mix to mono
            else:
                wav = wav.squeeze(0)
            return wav.numpy(), sr

        ref_wav, ref_sr = extract_wav(reference_audio)
        gen_wav, gen_sr = extract_wav(generated_audio)

        # 1. Speaker Similarity using speaker encoder from Base model
        speaker_encoder = self._load_speaker_encoder(speaker_encoder_model, local_model_path)

        ref_emb = self._extract_speaker_embedding(speaker_encoder, ref_wav, ref_sr)
        gen_emb = self._extract_speaker_embedding(speaker_encoder, gen_wav, gen_sr)

        speaker_sim = torch.nn.functional.cosine_similarity(
            ref_emb.flatten().unsqueeze(0),
            gen_emb.flatten().unsqueeze(0)
        ).item()

        # 2. Mel Spectrogram Distance
        target_sr = 24000
        if ref_sr != target_sr:
            ref_wav_mel = librosa.resample(ref_wav.astype(np.float32), orig_sr=ref_sr, target_sr=target_sr)
        else:
            ref_wav_mel = ref_wav
        if gen_sr != target_sr:
            gen_wav_mel = librosa.resample(gen_wav.astype(np.float32), orig_sr=gen_sr, target_sr=target_sr)
        else:
            gen_wav_mel = gen_wav

        with torch.no_grad():
            ref_mel = mel_spectrogram(
                torch.from_numpy(ref_wav_mel).unsqueeze(0),
                n_fft=1024, num_mels=128, sampling_rate=target_sr,
                hop_size=256, win_size=1024, fmin=0, fmax=12000
            )
            gen_mel = mel_spectrogram(
                torch.from_numpy(gen_wav_mel).unsqueeze(0),
                n_fft=1024, num_mels=128, sampling_rate=target_sr,
                hop_size=256, win_size=1024, fmin=0, fmax=12000
            )

            min_len = min(ref_mel.shape[-1], gen_mel.shape[-1])
            ref_mel = ref_mel[..., :min_len]
            gen_mel = gen_mel[..., :min_len]
            mel_mse = torch.nn.functional.mse_loss(ref_mel, gen_mel).item()

        # Determine quality rating
        if speaker_sim > 0.85:
            rating = "Excellent voice match"
        elif speaker_sim > 0.75:
            rating = "Good voice match"
        elif speaker_sim > 0.65:
            rating = "Moderate voice match"
        else:
            rating = "Poor voice match"

        # Calculate speaking rate
        ref_duration = len(ref_wav) / ref_sr
        gen_duration = len(gen_wav) / gen_sr
        rate_ratio = ref_duration / gen_duration

        if rate_ratio > 1.05:
            rate_desc = f"generated is {((rate_ratio - 1) * 100):.0f}% faster"
        elif rate_ratio < 0.95:
            rate_desc = f"generated is {((1 - rate_ratio) * 100):.0f}% slower"
        else:
            rate_desc = "similar pace"

        # Build report
        report = f"""Audio Comparison Report
========================
Speaker Similarity: {speaker_sim:.4f} (0-1, higher=better)
Mel Distance (MSE): {mel_mse:.6f} (lower=better)
Speaking Rate: {rate_ratio:.2f}x ({rate_desc})
Rating: {rating}

Interpretation Guide:
- Speaker Sim > 0.85: Excellent voice match
- Speaker Sim > 0.75: Good voice match
- Speaker Sim > 0.65: Moderate voice match
- Speaker Sim < 0.65: Poor voice match
- Speaking Rate ~1.0x: Ideal pacing match

Audio Details:
- Reference duration: {ref_duration:.2f}s
- Generated duration: {gen_duration:.2f}s"""

        print(report)
        return (report,)
