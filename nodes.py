import os
import json
import shutil
import torch
import contextlib
import io
import logging
import hashlib
from datetime import datetime, timezone
import soundfile as sf
import numpy as np
import folder_paths
import comfy.model_management as mm
from server import PromptServer
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
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
from safetensors.torch import save_file

# Register Qwen3-TTS models folder with ComfyUI
QWEN3_TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS")
os.makedirs(QWEN3_TTS_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS", QWEN3_TTS_MODELS_DIR)

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
    # waveform is [batch, channels, samples]
    # We need (samples,) or (channels, samples) numpy for Qwen?
    # Qwen accepts numpy array.
    
    if audio_input is None:
        return None
        
    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    
    # Take first batch item
    wav = waveform[0] # (channels, samples)
    
    # If multi-channel, maybe mix down or take first?
    # For cloning, mono is usually fine.
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0) # Mix to mono
    else:
        wav = wav.squeeze(0) # (samples,)
        
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
                "local_model_path": ("STRING", {"default": "", "multiline": False}),
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
            # MPS has limited bf16 support; fall back to fp16 on Mac
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        # Determine model path
        if local_model_path and local_model_path.strip() != "":
            model_path = local_model_path.strip()
            print(f"Loading from local path: {model_path}")
        else:
            # Check if model exists in ComfyUI models folder, download if not
            local_path = get_local_model_path(repo_id)
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"Loading from ComfyUI models folder: {model_path}")
            else:
                # Download only this specific model
                print(f"Model not found locally. Downloading {repo_id}...")
                model_path = download_model_to_comfyui(repo_id, source)

        print(f"Loading Qwen3-TTS model: {repo_id} from {model_path} on {device} as {dtype}")
        
        # Determine attention implementation
        attn_impl = "sdpa" # Default to sdpa (torch 2.0+) as it's usually available and fast
        
        if attention != "auto":
            attn_impl = attention
        else:
            # Auto-detect
            try:
                import flash_attn
                # Also check version metadata as transformers works 
                import importlib.metadata
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except Exception:
                # Fallback to sdpa if flash_attn missing or metadata broken
                attn_impl = "sdpa"

        print(f"Using attention implementation: {attn_impl}")

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )
        
        # FORCE SPEAKER MAPPING FIX - Deep Injection
        try:
            cfg_file = os.path.join(model_path, "config.json")
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
                "max_new_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "step": 64}),
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
            }
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"

    def create_prompt(self, model, ref_audio, ref_text):
        audio_tuple = load_audio_input(ref_audio)
        
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
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, seed, language="Auto", ref_audio=None, ref_text=None, prompt=None):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, seed, language="Auto", ref_audio=None, ref_text=None, prompt=None):
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
                    voice_clone_prompt=prompt
                )
            elif ref_audio is not None and ref_text is not None and ref_text.strip() != "":
                # Use on-the-fly prompt creation
                audio_tuple = load_audio_input(ref_audio)
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=audio_tuple,
                    ref_text=ref_text
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
                 "keep_intermediate_checkpoints": ("BOOLEAN", {"default": False, "tooltip": "Keep all epoch checkpoints. When False, only the latest checkpoint is kept to save disk space."}),
                 # VRAM Optimizations
                 "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16", "tooltip": "Mixed precision training mode. bf16 recommended for modern GPUs."}),
                 "gradient_accumulation": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Accumulate gradients over N steps before updating. Effective batch size = batch_size * gradient_accumulation."}),
                 "gradient_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "Trade compute for VRAM by recomputing activations. Saves ~30-40% VRAM."}),
                 "use_8bit_optimizer": ("BOOLEAN", {"default": True, "tooltip": "Use 8-bit AdamW optimizer. Saves ~50% optimizer VRAM. Requires bitsandbytes."}),
                 # Training Dynamics
                 "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "L2 regularization strength to prevent overfitting."}),
                 "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Gradient clipping threshold to prevent exploding gradients."}),
                 # Learning Rate Schedule
                 "warmup_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Number of warmup steps. Set to 0 to disable warmup. Recommended: 5-10% of total steps."}),
                 "warmup_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Warmup as ratio of total steps. Ignored if warmup_steps > 0. E.g., 0.1 = 10% warmup."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "custom_speaker_name")
    FUNCTION = "train"
    CATEGORY = "Qwen3-TTS/FineTuning"

    def train(self, train_jsonl, init_model, source, output_dir, epochs, batch_size, lr, speaker_name, seed, mixed_precision="bf16", resume_training=False, keep_intermediate_checkpoints=False, gradient_accumulation=4, gradient_checkpointing=True, use_8bit_optimizer=True, weight_decay=0.01, max_grad_norm=1.0, warmup_steps=0, warmup_ratio=0.0, unique_id=None):
        # Helper to send progress text to UI
        def send_status(text):
            if unique_id:
                PromptServer.instance.send_progress_text(text, unique_id)

        # Setup output directory
        full_output_dir = os.path.abspath(output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        # Check for existing checkpoints if resume is enabled
        start_epoch = 0
        resume_checkpoint_path = None
        if resume_training:
            # Find latest checkpoint in output_dir
            checkpoints = []
            for item in os.listdir(full_output_dir) if os.path.exists(full_output_dir) else []:
                if item.startswith("epoch_"):
                    try:
                        epoch_num = int(item.split("_")[1])
                        checkpoint_path = os.path.join(full_output_dir, item)
                        if os.path.isdir(checkpoint_path):
                            checkpoints.append((epoch_num, checkpoint_path))
                    except (ValueError, IndexError):
                        pass
            if checkpoints:
                checkpoints.sort(key=lambda x: x[0], reverse=True)
                start_epoch = checkpoints[0][0]
                resume_checkpoint_path = checkpoints[0][1]
                print(f"Resume enabled: Found checkpoint at epoch {start_epoch}")
                print(f"Will continue training from epoch {start_epoch + 1} to {start_epoch + epochs}")

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

                # Accelerator setup - respect ComfyUI's --cpu flag
                # Effective batch size = batch_size * gradient_accumulation (default: 4)
                use_cpu = mm.cpu_mode()
                accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation, mixed_precision=mixed_precision, cpu=use_cpu)

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
                except:
                     pass

                print(f"Using attention implementation: {attn_impl}")

                dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16 if mixed_precision == "fp16" else torch.float32

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

                # Calculate total training steps for THIS run
                num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation
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

                # Calculate total steps and logging interval
                total_steps_per_epoch = len(train_dataloader)
                log_interval = max(1, total_steps_per_epoch // 10)

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
                            
                            speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                            if target_speaker_embedding is None:
                                target_speaker_embedding = speaker_embedding
                                
                            input_text_ids = input_ids[:, :, 0]
                            input_codec_ids = input_ids[:, :, 1]
                            
                            # Use model directly (accelerator unwraps attributes automatically usually)
                            # If model is DDP, it might fail, but for single GPU Comfy it should pass attributes.
                            current_model = model
                            
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

                            # Show step progress periodically
                            if steps % log_interval == 0 or steps == total_steps_per_epoch:
                                avg_loss_so_far = epoch_loss / steps
                                status = f"Epoch {epoch + 1}/{end_epoch} - Step {steps}/{total_steps_per_epoch} - Loss: {avg_loss_so_far:.4f}"
                                print(status)
                                send_status(status)

                    avg_loss = epoch_loss/steps if steps > 0 else 0
                    print(f"Epoch {epoch + 1}/{end_epoch} - Avg Loss: {avg_loss}")
                    send_status(f"Epoch {epoch + 1}/{end_epoch} - Loss: {avg_loss:.4f}")

                    # Save checkpoint after each epoch
                    send_status(f"Saving checkpoint epoch {epoch + 1}...")
                    checkpoint_path = os.path.join(full_output_dir, f"epoch_{epoch + 1}")
                    os.makedirs(checkpoint_path, exist_ok=True)

                    unwrapped_model_ckpt = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model_ckpt.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))

                    # Save optimizer state for resume (preserves momentum/Adam statistics)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

                    # Save scheduler state for resume
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

                    # Suppress verbose library output during save (stdout, stderr, and logging)
                    prev_log_level = logging.root.manager.disable
                    logging.disable(logging.CRITICAL)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            config.save_pretrained(checkpoint_path)
                            qwen3tts.processor.save_pretrained(checkpoint_path)
                    finally:
                        logging.disable(prev_log_level)

                    # Modify checkpoint config for custom voice
                    ckpt_config_path = os.path.join(checkpoint_path, "config.json")
                    with open(ckpt_config_path, 'r', encoding='utf-8') as f:
                        ckpt_config_dict = json.load(f)

                    # Sanitize problematic model_type keys
                    keys_to_sanitize = ["speaker_encoder_config", "decoder_config", "encoder_config"]
                    for key in keys_to_sanitize:
                        if key in ckpt_config_dict and isinstance(ckpt_config_dict[key], dict):
                            if "model_type" in ckpt_config_dict[key]:
                                del ckpt_config_dict[key]["model_type"]

                    # Set custom voice type
                    ckpt_config_dict["tts_model_type"] = "custom_voice"

                    # Add speaker config
                    speaker_name_key = speaker_name.lower()
                    cfg = ckpt_config_dict.get("talker_config", {})
                    if not isinstance(cfg, dict):
                        cfg = {}

                    spk_id = cfg.get("spk_id", {})
                    if not isinstance(spk_id, dict): spk_id = {}
                    spk_id[speaker_name_key] = 3000
                    cfg["spk_id"] = spk_id

                    spk_is_dialect = cfg.get("spk_is_dialect", {})
                    if not isinstance(spk_is_dialect, dict): spk_is_dialect = {}
                    spk_is_dialect[speaker_name_key] = False
                    cfg["spk_is_dialect"] = spk_is_dialect

                    ckpt_config_dict["talker_config"] = cfg

                    with open(ckpt_config_path, 'w', encoding='utf-8') as f:
                        json.dump(ckpt_config_dict, f, indent=2, ensure_ascii=False)

                    # Copy speech_tokenizer for each checkpoint
                    st_source_ckpt = None
                    base_model_for_st = init_model_path if not resume_checkpoint_path else resume_checkpoint_path
                    if os.path.isdir(base_model_for_st):
                        local_st_ckpt = os.path.join(base_model_for_st, "speech_tokenizer")
                        if os.path.isdir(local_st_ckpt):
                            st_source_ckpt = local_st_ckpt
                    if st_source_ckpt:
                        target_st_ckpt = os.path.join(checkpoint_path, "speech_tokenizer")
                        if not os.path.exists(target_st_ckpt):
                            shutil.copytree(st_source_ckpt, target_st_ckpt)

                    print(f"Checkpoint saved: {checkpoint_path}")

                    # Delete previous epoch checkpoint to save space (unless user wants to keep them)
                    if not keep_intermediate_checkpoints and epoch > start_epoch:
                        prev_checkpoint_path = os.path.join(full_output_dir, f"epoch_{epoch}")
                        # Safety check: only delete if it looks like a checkpoint we created
                        is_valid_checkpoint = (
                            os.path.isdir(prev_checkpoint_path) and
                            os.path.exists(os.path.join(prev_checkpoint_path, "pytorch_model.bin")) and
                            os.path.exists(os.path.join(prev_checkpoint_path, "config.json"))
                        )
                        if is_valid_checkpoint:
                            shutil.rmtree(prev_checkpoint_path)
                            print(f"Deleted previous checkpoint: {prev_checkpoint_path}")
                        elif os.path.exists(prev_checkpoint_path):
                            print(f"Skipped deletion of {prev_checkpoint_path}: not a valid checkpoint directory")

                # Save final model (same as last epoch checkpoint)
                final_output_path = os.path.join(full_output_dir, f"epoch_{end_epoch}")
                os.makedirs(final_output_path, exist_ok=True)
                
                print("Saving trained model...")
                unwrapped_model = accelerator.unwrap_model(model)
                
                # We skip unwrapped_model.save_pretrained because it fails on config diffs (KeyError: 'dtype')
                # We save config manually instead (suppress verbose library output)
                prev_log_level = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        config.save_pretrained(final_output_path)
                        qwen3tts.processor.save_pretrained(final_output_path)
                finally:
                    logging.disable(prev_log_level)
                
                # Save speech tokenizer which is required for loading
                # Try to copy it from the source model location
                st_source = None
                if os.path.isdir(init_model_path):
                     local_st = os.path.join(init_model_path, "speech_tokenizer")
                     if os.path.isdir(local_st):
                         st_source = local_st
                else:
                    # Try HF Cache
                    try:
                        st_config = cached_file(init_model_path, "speech_tokenizer/config.json")
                        if st_config:
                            st_source = os.path.dirname(st_config)
                    except:
                        pass
                        
                if st_source:
                    target_st = os.path.join(final_output_path, "speech_tokenizer")
                    if os.path.exists(target_st):
                        shutil.rmtree(target_st)
                    shutil.copytree(st_source, target_st)
                    print(f"Copied speech_tokenizer from {st_source}")
                else:
                    print("WARNING: Could not find speech_tokenizer to copy. Loading this model might fail!")

                # Copy generation_config.json if it exists in source
                gen_config_source = None
                if os.path.isdir(init_model_path):
                    local_gen = os.path.join(init_model_path, "generation_config.json")
                    if os.path.exists(local_gen):
                        gen_config_source = local_gen
                else:
                    try:
                        gen_config_source = cached_file(init_model_path, "generation_config.json")
                    except:
                        pass
                
                if gen_config_source:
                    shutil.copy2(gen_config_source, os.path.join(final_output_path, "generation_config.json"))
                    print(f"Copied generation_config from {gen_config_source}")

                # Modify Config for Custom Voice
                config_path = os.path.join(final_output_path, "config.json")
                with open(config_path, 'r', encoding='utf-8') as f:
                     config_dict = json.load(f)
                
                # Sanitize the config of any "model_type" keys in nested configs which cause TypeError on load
                # NOTE: We MUST NOT sanitize "talker_config" because it needs its model_type to load correctly.
                # speaker_encoder_config is known to crash if model_type is present.
                keys_to_sanitize = ["speaker_encoder_config", "decoder_config", "encoder_config"]
                for key in keys_to_sanitize:
                    if key in config_dict and isinstance(config_dict[key], dict):
                         if "model_type" in config_dict[key]:
                             print(f"Sanitizing {key}: removing model_type")
                             del config_dict[key]["model_type"]

                config_dict["tts_model_type"] = "custom_voice"
                
                # FORCE LOWERCASE name for strict library compatibility
                speaker_name_key = speaker_name.lower()

                # We only update talker_config. speaker_encoder_config is strict and lacks spk_id/spk_is_dialect.
                cfg_key = "talker_config"
                cfg = config_dict.get(cfg_key, {})
                if not isinstance(cfg, dict):
                    cfg = {}
                
                spk_id = cfg.get("spk_id", {})
                if not isinstance(spk_id, dict): spk_id = {}
                spk_id[speaker_name_key] = 3000
                cfg["spk_id"] = spk_id
                
                spk_is_dialect = cfg.get("spk_is_dialect", {})
                if not isinstance(spk_is_dialect, dict): spk_is_dialect = {}
                spk_is_dialect[speaker_name_key] = False
                cfg["spk_is_dialect"] = spk_is_dialect
                
                config_dict[cfg_key] = cfg
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                # Save training checkpoint (includes all weights for resume)
                torch.save(unwrapped_model.state_dict(), os.path.join(final_output_path, "pytorch_model.bin"))
                print(f"Saved training checkpoint: {os.path.join(final_output_path, 'pytorch_model.bin')}")

                # Save optimizer state for resume (preserves momentum/Adam statistics)
                torch.save(optimizer.state_dict(), os.path.join(final_output_path, "optimizer.pt"))

                # Save scheduler state
                if scheduler:
                    torch.save(scheduler.state_dict(), os.path.join(final_output_path, "scheduler.pt"))

                # Save specific weights for speaker embedding (injecting into index 3000)
                state_dict = unwrapped_model.state_dict()
                state_dict = {k: v.cpu() for k, v in state_dict.items()}
                
                drop_prefix = "speaker_encoder"
                keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                for k in keys_to_drop:
                     del state_dict[k]
                
                if target_speaker_embedding is not None:
                     weight = state_dict['talker.model.codec_embedding.weight']
                     state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().cpu().to(weight.dtype)

                save_file(state_dict, os.path.join(final_output_path, "model.safetensors"))

                # Cleanup: free accelerator resources and synchronize CUDA
                accelerator.free_memory()
                del model, optimizer, train_dataloader, qwen3tts, unwrapped_model
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                print(f"Fine-tuning complete. Model saved to {final_output_path}")
                send_status("Training complete!")
                return (final_output_path, speaker_name)


class Qwen3AudioCompare:
    # Class-level cache for speaker encoder
    _speaker_encoder = None
    _speaker_encoder_model = None

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
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "compare"
    CATEGORY = "Qwen3-TTS/Evaluation"

    def _load_speaker_encoder(self, model_repo):
        """Load only the speaker encoder from a Base model (not the full model)."""
        from safetensors.torch import load_file
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

        # Check if already cached
        if Qwen3AudioCompare._speaker_encoder is not None and Qwen3AudioCompare._speaker_encoder_model == model_repo:
            return Qwen3AudioCompare._speaker_encoder

        # Get local model path
        model_path = get_local_model_path(model_repo)
        if not os.path.exists(model_path):
            raise ValueError(f"Base model not found at {model_path}. Please download it first using Qwen3-TTS Loader.")

        # Load config to get speaker encoder config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Create speaker encoder config
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        speaker_config = Qwen3TTSSpeakerEncoderConfig(**config_dict["speaker_encoder_config"])

        # Instantiate speaker encoder
        speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config)

        # Load only speaker encoder weights from safetensors
        safetensors_path = os.path.join(model_path, "model.safetensors")
        all_weights = load_file(safetensors_path)

        # Filter to only speaker_encoder weights and remove prefix
        speaker_weights = {}
        for k, v in all_weights.items():
            if k.startswith("speaker_encoder."):
                new_key = k[len("speaker_encoder."):]
                speaker_weights[new_key] = v

        speaker_encoder.load_state_dict(speaker_weights)
        speaker_encoder.eval()

        # Move to GPU if available
        device = mm.get_torch_device()
        speaker_encoder = speaker_encoder.to(device)

        # Cache it
        Qwen3AudioCompare._speaker_encoder = speaker_encoder
        Qwen3AudioCompare._speaker_encoder_model = model_repo

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

    def compare(self, reference_audio, generated_audio, speaker_encoder_model):
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
        speaker_encoder = self._load_speaker_encoder(speaker_encoder_model)

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
