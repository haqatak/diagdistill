import argparse
import os
import re
import urllib.request
import time
from datetime import datetime
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
import peft

import gradio as gr

from pipeline import CausalInferencePipeline
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.misc import set_seed
from utils.lora_utils import configure_lora_for_model
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

class AppConfig:
    def __init__(self, config_path):
        self.config = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_memory = get_cuda_free_memory_gb(self.device) < 40
        self.pipeline = None
        self.is_loaded = False

    def initialize_vae_decoder(self, use_taehv: bool, taehv_checkpoint_path: str, device: torch.device):
        if not use_taehv:
            return None

        from taehv import TAEHV

        if not os.path.exists(taehv_checkpoint_path):
            print(f"taew2_1.pth not found at {taehv_checkpoint_path}. Downloading...")
            os.makedirs(os.path.dirname(taehv_checkpoint_path), exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
            print(f"Downloaded TAEHV checkpoint to {taehv_checkpoint_path}")

        class TinyVAEWrapper(torch.nn.Module):
            def __init__(self, checkpoint_path: str):
                super().__init__()
                self.taehv = TAEHV(checkpoint_path=checkpoint_path).to(torch.float16)

            def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
                del use_cache
                latents_fp16 = latent.to(dtype=torch.float16)
                return self.taehv.decode_video(latents_fp16, parallel=False).mul_(2).sub_(1)

        tiny_vae = TinyVAEWrapper(taehv_checkpoint_path).eval().requires_grad_(False)
        tiny_vae.to(device)
        print("TAEHV tiny decoder enabled for inference")
        return tiny_vae

    def load_model(self):
        if self.is_loaded:
            return

        print("Initializing pipeline...")
        torch.set_grad_enabled(False)
        self.pipeline = CausalInferencePipeline(self.config, device=self.device)

        if self.config.generator_ckpt:
            state_dict = torch.load(self.config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "generator_ema" in state_dict:
                raw_gen_state_dict = state_dict["generator_ema" if self.config.get("use_ema", False) else "generator"]
            elif "model" in state_dict:
                raw_gen_state_dict = state_dict["model"]
            else:
                raise ValueError(f"Generator state dict not found in {self.config.generator_ckpt}")

            if self.config.get("use_ema", False):
                def _clean_key(name: str) -> str:
                    name = name.replace("_fsdp_wrapped_module.", "")
                    return name

                cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
                missing, unexpected = self.pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters missing")
            else:
                self.pipeline.generator.load_state_dict(raw_gen_state_dict)

        self.pipeline.is_lora_enabled = False
        if getattr(self.config, "adapter", None) and configure_lora_for_model is not None:
            print("Applying LoRA to generator (inference)...")
            self.pipeline.generator.model = configure_lora_for_model(
                self.pipeline.generator.model,
                model_name="generator",
                lora_config=self.config.adapter,
                is_main_process=True,
            )

            lora_ckpt_path = getattr(self.config, "lora_ckpt", None)
            if lora_ckpt_path:
                print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
                lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                    peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint["generator_lora"])
                else:
                    peft.set_peft_model_state_dict(self.pipeline.generator.model, lora_checkpoint)
                print("LoRA weights loaded for generator")
            self.pipeline.is_lora_enabled = True

        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)

        if getattr(self.config, "use_fp8", False):
            print("Applying FP8 quantization to generator...")
            quantize_(self.pipeline.generator, float8_dynamic_activation_float8_weight())

        if self.low_memory:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=self.device)

        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)

        use_taehv = bool(getattr(self.config, "use_taehv", False))
        taehv_checkpoint_path = str(getattr(self.config, "taehv_checkpoint_path", "checkpoints/taew2_1.pth"))
        if use_taehv:
            tiny_vae = self.initialize_vae_decoder(use_taehv=True, taehv_checkpoint_path=taehv_checkpoint_path, device=self.device)
            if tiny_vae is not None:
                self.pipeline.vae = tiny_vae

        if bool(getattr(self.config, "enable_torch_compile", False)):
            compile_mode = str(getattr(self.config, "torch_compile_mode", "max-autotune-no-cudagraphs"))
            print(f"Enabling torch.compile with mode={compile_mode}")
            try:
                compiled_gen = torch.compile(self.pipeline.generator, mode=compile_mode)
                if compiled_gen is not None:
                    self.pipeline.generator = compiled_gen
            except Exception as e:
                print(f"[Warning] torch.compile failed for generator: {e}")
            try:
                compiled_vae = torch.compile(self.pipeline.vae, mode=compile_mode)
                if compiled_vae is not None:
                    self.pipeline.vae = compiled_vae
            except Exception as e:
                print(f"[Warning] torch.compile failed for VAE: {e}")

        print("Model loaded successfully.")
        self.is_loaded = True

def run_gradio(config_path):
    app_config = AppConfig(config_path)

    def generate_video(prompt, seed):
        if not app_config.is_loaded:
            try:
                app_config.load_model()
            except Exception as e:
                return None, f"Error loading model: {e}"

        set_seed(int(seed))

        output_folder = "videos/gradio_output"
        os.makedirs(output_folder, exist_ok=True)

        num_samples = 1
        prompts = [prompt] * num_samples
        num_output_frames = app_config.config.num_output_frames

        sampled_noise = torch.randn(
            [num_samples, num_output_frames, 16, 60, 104], device=app_config.device, dtype=torch.bfloat16
        )

        try:
            video, latents = app_config.pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                low_memory=app_config.low_memory,
                profile=False,
            )
        except Exception as e:
            return None, f"Error during inference: {e}"

        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_tensor = 255.0 * current_video

        if hasattr(app_config.pipeline.vae, "model") and hasattr(app_config.pipeline.vae.model, "clear_cache"):
            app_config.pipeline.vae.model.clear_cache()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_stem = re.sub(r"\s+", "_", prompt[:50].strip()).replace("/", "_").replace("\\", "_")
        if not prompt_stem:
            prompt_stem = "empty_prompt"

        output_path = os.path.join(output_folder, f'{prompt_stem}_{timestamp}.mp4')

        try:
            write_video(output_path, video_tensor[0], fps=16)
            return output_path, "Generation complete."
        except Exception as e:
            return None, f"Error saving video: {e}"

    with gr.Blocks(title="Diagonal Distillation Video Generation") as demo:
        gr.Markdown("# Diagonal Distillation Video Generation")
        gr.Markdown("Generate high-quality videos quickly using Diagonal Distillation.")

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your text prompt here...", lines=4)
                seed_input = gr.Number(label="Seed", value=0, precision=0)
                generate_button = gr.Button("Generate Video", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                video_output = gr.Video(label="Generated Video")

        generate_button.click(
            fn=generate_video,
            inputs=[prompt_input, seed_input],
            outputs=[video_output, status_output]
        )

    demo.queue().launch(share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/diadistill_inference.yaml", help="Path to the config file")
    args = parser.parse_args()
    run_gradio(args.config_path)
