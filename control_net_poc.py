from diffusers.utils import load_image
from PIL import Image
import numpy as np 

from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

from diffusers import  ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetImg2ImgPipeline
import torch
import torchvision

import json
import os

import numpy as np

# torch.cuda.set_device(3)

with open("./Filtered.json", "rb") as f:
    prompt_data = json.load(fp=f)

print(prompt_data['0']['Image ID'])

height = 695
width = 248

images = []
prompts = []
negative_prompts =[ "cartoonish, unrealistic, bad anatomy, worst quality, low quality"]

cond_pose_images = []

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

dataset_address_prefix = "./IIITD-20K/"


for i in range(1):

    image_path = ""
    
    jpeg_url = f"{dataset_address_prefix}"+prompt_data[str(i)]['Image ID']+".jpeg"
    jpg_url = f"{dataset_address_prefix}"+prompt_data[str(i)]['Image ID']+".jpg"

    if os.path.exists(jpeg_url):
      image_path = jpeg_url
    elif os.path.exists(jpg_url):
      image_path = jpg_url
    else:
      print("WHATT ")
      print(jpeg_url)
      continue

    image = Image.open(image_path)
    image = image.resize((width, height))

    openpose_image = openpose(image).resize((width, height))

    images.append(image)
    cond_pose_images.append(openpose_image)
    prompts.append(prompt_data[str(i)]['Description 1'])



controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
)
model_id = "runwayml/stable-diffusion-v1-5"

generator = torch.manual_seed(0)


pipe_I2I = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
)

pipe_I2I.scheduler = UniPCMultistepScheduler.from_config(pipe_I2I.scheduler.config)
pipe_I2I.enable_model_cpu_offload()


NUM_GENERATIONS = 2

for i in range(1):

    gen_images = pipe_I2I(prompt=[prompts[i]]*NUM_GENERATIONS, negative_prompt=negative_prompts * NUM_GENERATIONS, control_image=cond_pose_images[i], image=images[i] ,generator=generator, height=height, width=width)

    gen_images = [torch.tensor(np.array(im)) for im in gen_images.images]
    gen_images = [ torch.permute(im, (2, 0, 1)).unsqueeze(0) for im in gen_images]
    print(gen_images[0].shape)
    print(len(gen_images))
    prompt_image = torch.permute(torch.tensor(np.array(images[i].resize((gen_images[-1].shape[-1], gen_images[-1].shape[-2])))), (2,0,1)).unsqueeze(0)
    pose_image = torch.permute(torch.tensor(np.array(cond_pose_images[i].resize((gen_images[-1].shape[-1], gen_images[-1].shape[-2])))), (2,0,1)).unsqueeze(0)
    gen_images.extend([prompt_image, pose_image])

    print([im.shape for im in gen_images])

    # torchvision.utils.save_image(torch.cat(gen_images), f"./vis/{i}.png")



    grid = torchvision.utils.make_grid(torch.cat(gen_images))
    ndarr = grid
    # ndarr = ndarr.add_(0.5).clamp_(0, 255)
    ndarr = ndarr.to("cpu", torch.uint8).numpy()
    ndarr = ndarr.transpose((1,2,0))
    print(type(ndarr), ndarr.shape)
    im = Image.fromarray(ndarr)
    im.save(f"./vis/{i}.png")

    

    


# # images.images[0].show()