import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
# from diffusers import StableDiffusionControlNetImg2ImgPipeline
from pose_gen.pipeline.pipeline_controlnet_img2img_temporal_attention import StableDiffusionControlNetImg2ImgTemporalAttnPipeline
from pose_gen.pipeline.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline


def make_pipeline(generator_seed: int = 0):
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
    )
    model_id = "runwayml/stable-diffusion-v1-5"

    generator = torch.manual_seed(generator_seed)

#     pipe_I2I = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#         model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
#     )

    pipe_I2I = StableDiffusionControlNetImg2ImgTemporalAttnPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )

    pipe_I2I.scheduler = UniPCMultistepScheduler.from_config(pipe_I2I.scheduler.config)
    pipe_I2I.enable_model_cpu_offload()

    return pipe_I2I, generator


make_pipeline()
