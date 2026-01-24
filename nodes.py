import os
import json
import shutil
import torch
import soundfile as sf
import numpy as np
import folder_paths
import comfy.model_management as mm
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from .dataset import TTSDataset
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.utils import cached_file
from safetensors.torch import save_file

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
                "repo_id": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                ], {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
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
            
        model_path = repo_id
        
        if local_model_path and local_model_path.strip() != "":
            model_path = local_model_path.strip()
            print(f"Loading from local path: {model_path}")
        elif source == "ModelScope":
            from modelscope import snapshot_download
            try:
                model_path = snapshot_download(repo_id)
            except Exception as e:
                print(f"ModelScope download failed: {e}. Falling back to HF or local.")

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
            }
        }
    
    @classmethod
    def IS_CHANGED(s, model, text, language, speaker, seed, instruct="", custom_speaker_name=""):
        return seed

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, language, speaker, seed, instruct="", custom_speaker_name=""):
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
                instruct=inst
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
                "tokenizer_repo": ("STRING", {"default": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_jsonl_path",)
    FUNCTION = "process"
    CATEGORY = "Qwen3-TTS/FineTuning"

    def process(self, jsonl_path, tokenizer_repo):
        device = mm.get_torch_device()
        
        output_path = jsonl_path.replace(".jsonl", "_codes.jsonl")
        
        print(f"Loading Tokenizer: {tokenizer_repo}")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_repo,
            device_map=device,
        )
        
        inputs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                inputs.append(json.loads(line.strip()))
                
        batch_size = 32
        final_lines = []
        
        print(f"Processing {len(inputs)} items...")
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            audio_paths = [b['audio'] for b in batch]
            
            # Encode
            # If batch has 1 item, encode returns object. If list, might behave differently?
            # Qwen tokenizer handles list of paths.
            enc_res = tokenizer.encode(audio_paths)
            
            # encode returns an object with audio_codes attribute
            codes = enc_res.audio_codes # tensor
            
            for j, code in enumerate(codes):
                item = batch[j]
                item['audio_codes'] = code.cpu().tolist()
                final_lines.append(item)
                
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in final_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                
        print(f"Processed dataset saved to {output_path}")
        return (output_path,)

class Qwen3FineTune:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "train_jsonl": ("STRING", {"default": "", "multiline": False}),
                "init_model": ("STRING", {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "multiline": False}),
                "output_dir": ("STRING", {"default": "output/finetuned_model", "multiline": False}),
                "epochs": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 64}),
                "lr": ("FLOAT", {"default": 2e-5, "step": 1e-6}),
                "speaker_name": ("STRING", {"default": "my_speaker"}),
            },
            "optional": {
                 "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "train"
    CATEGORY = "Qwen3-TTS/FineTuning"

    def train(self, train_jsonl, init_model, output_dir, epochs, batch_size, lr, speaker_name, mixed_precision):
        # Setup output directory
        full_output_dir = os.path.abspath(output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # ComfyUI runs in inference_mode by default. 
        # We must disable it and enable gradients properly for the entire scope, including model loading.
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                # Accelerator setup - respect ComfyUI's --cpu flag
                use_cpu = mm.cpu_mode()
                accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision=mixed_precision, cpu=use_cpu)
                
                print(f"Loading base model: {init_model}")
                
                attn_impl = "sdpa"
                try:
                     import flash_attn
                     import importlib.metadata
                     importlib.metadata.version("flash_attn")
                     attn_impl = "flash_attention_2"
                except:
                     pass

                dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16 if mixed_precision == "fp16" else torch.float32

                qwen3tts = Qwen3TTSModel.from_pretrained(
                    init_model,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                )
                
                # FORCE GRADIENTS ON
                qwen3tts.model.train()
                for name, param in qwen3tts.model.named_parameters():
                    param.requires_grad = True
                
                config = AutoConfig.from_pretrained(init_model)
                
                # Load Data
                with open(train_jsonl, 'r', encoding='utf-8') as f:
                    train_lines = [json.loads(line) for line in f]
                    
                dataset = TTSDataset(train_lines, qwen3tts.processor, config)
                train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
                
                optimizer = AdamW(qwen3tts.model.parameters(), lr=lr, weight_decay=0.01)
                
                model, optimizer, train_dataloader = accelerator.prepare(
                    qwen3tts.model, optimizer, train_dataloader
                )
                
                model.train()
                
                target_speaker_embedding = None
                
                print(f"Starting training for {epochs} epochs...")
                
                for epoch in range(epochs):
                    epoch_loss = 0
                    steps = 0
                    for batch in train_dataloader:
                        with accelerator.accumulate(model):
                            # Debug info
                            if steps == 0 and epoch == 0:
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
                            if steps == 0 and epoch == 0:
                                print(f"DEBUG: Model Training Mode: {current_model.training}")
                                # Check embedding layer grad
                                emb_layer = current_model.talker.model.text_embedding
                                print(f"DEBUG: Text Embedding Layer Weight requires_grad: {emb_layer.weight.requires_grad}")

                            input_text_embedding = current_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                            input_codec_embedding = current_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                            input_codec_embedding[:, 6, :] = speaker_embedding
                            
                            input_embeddings = input_text_embedding + input_codec_embedding
                            
                            if steps == 0 and epoch == 0:
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
                            
                            if steps == 0 and epoch == 0:
                                print(f"DEBUG: Loss requires_grad: {loss.requires_grad}")
                                if not loss.requires_grad:
                                    print(f"DEBUG: outputs.loss requires_grad: {outputs.loss.requires_grad if outputs.loss is not None else 'None'}")
                                    print(f"DEBUG: sub_talker_loss requires_grad: {sub_talker_loss.requires_grad}")
                            
                            accelerator.backward(loss)
                            
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                                
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            epoch_loss += loss.item()
                            steps += 1
                    
                    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/steps if steps>0 else 0}")
                    
                # Save Model (Last Epoch)
                final_output_path = os.path.join(full_output_dir, f"epoch_{epochs}")
                os.makedirs(final_output_path, exist_ok=True)
                
                print("Saving trained model...")
                unwrapped_model = accelerator.unwrap_model(model)
                
                # We skip unwrapped_model.save_pretrained because it fails on config diffs (KeyError: 'dtype')
                # We save config manualy instead
                config.save_pretrained(final_output_path)
                qwen3tts.processor.save_pretrained(final_output_path)
                
                # Save speech tokenizer which is required for loading
                # Try to copy it from the source model location
                st_source = None
                if os.path.isdir(init_model):
                     local_st = os.path.join(init_model, "speech_tokenizer")
                     if os.path.isdir(local_st):
                         st_source = local_st
                else:
                    # Try HF Cache
                    try:
                        st_config = cached_file(init_model, "speech_tokenizer/config.json")
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
                if os.path.isdir(init_model):
                    local_gen = os.path.join(init_model, "generation_config.json")
                    if os.path.exists(local_gen):
                        gen_config_source = local_gen
                else:
                    try:
                        gen_config_source = cached_file(init_model, "generation_config.json")
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
                    
                # Save specific weights for speaker embedding (injecting into index 3000)
                state_dict = unwrapped_model.state_dict()
                state_dict = {k: v.cpu() for k, v in state_dict.items()}
                
                drop_prefix = "speaker_encoder"
                keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                for k in keys_to_drop:
                     del state_dict[k]
                
                if target_speaker_embedding is not None:
                     weight = state_dict['talker.model.codec_embedding.weight']
                     state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.dtype)
                
                save_file(state_dict, os.path.join(final_output_path, "model.safetensors"))
        
                print(f"Fine-tuning complete. Model saved to {final_output_path}")
                return (final_output_path,)
