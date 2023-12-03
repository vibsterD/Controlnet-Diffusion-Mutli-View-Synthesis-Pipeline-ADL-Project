from diffusers.utils import load_image
from PIL import Image
import numpy as np

# from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

from diffusers import ControlNetModel, UniPCMultistepScheduler
# from diffusers import StableDiffusionControlNetImg2ImgPipeline
from pipeline.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
import torch
import torchvision

import json
import os

import numpy as np

# IIITD_DATASET_PREFIX = "/raid/home/vibhu20150/Datasets/IIITD-20K/"
IIITD_DATASET_PREFIX = "./data/IIITD-20K/"
SAVE_DIR = "./360_vid_images_mac/"
# SAVE_DIR = "./360_vid_images/"

def make_pipeline(generator_seed: int = 0):
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
    )
    model_id = "runwayml/stable-diffusion-v1-5"

    generator = torch.manual_seed(generator_seed)

    pipe_I2I = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )

    pipe_I2I.scheduler = UniPCMultistepScheduler.from_config(pipe_I2I.scheduler.config)
    pipe_I2I.enable_model_cpu_offload()

    return pipe_I2I, generator


def load_IIITD_dataset_json():
    with open(IIITD_DATASET_PREFIX + "Filtered.json", "rb") as f:
        prompt_data = json.load(fp=f)

    return prompt_data


def load_pose_frames_360(height: int, width: int):
    pose_images_cropped = []
    pose_image_paths = os.listdir("./360-cropped-pose-data/")
    pose_image_paths.sort()

    if len(pose_image_paths) > 0:
        for path in pose_image_paths:
            pose_images_cropped.append(Image.open("./360-cropped-pose-data/" + path))
        print(np.array(pose_images_cropped[0]).shape)
        return pose_images_cropped

    pose_image_paths = os.listdir("./data")
    pose_image_paths.sort()

    for path in pose_image_paths:
        pose_image = np.array(Image.open("./data/" + path))
        pose_image_cropped = pose_image[100:-50, 550:-550, :]
        crp_img = Image.fromarray(pose_image_cropped)
        crp_img = crp_img.resize((width, height))
        crp_img.save(f"./360-cropped-pose-data/{path}_cropped.png")
        pose_images_cropped.append(crp_img)

    return pose_images_cropped


def paste_images_horizontally(images: list[Image.Image]) -> Image.Image:
    if len(images) < 1:
        raise ValueError("images list should contain more than one image")

    width = 0
    height = 0
    for im in images:
        height = max(height, im.size[1])
        width += im.size[0]

    new_image = Image.new(images[0].mode, (width, height))

    x_offset = 0

    for im in images:
        new_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_image.save(SAVE_DIR + 'test.png')

    return new_image


def main():
    prompt_data = load_IIITD_dataset_json()

    print("First image ID:", prompt_data["0"]["Image ID"])

    NUM_FRAMES_PER_GEN = 3
    NUM_IMAGES_TO_PARSE = 1


    HEIGHT = 695
    WIDTH = 249*NUM_FRAMES_PER_GEN

    assert WIDTH%NUM_FRAMES_PER_GEN == 0

    images = []
    cond_pose_images = load_pose_frames_360(HEIGHT, WIDTH)
    prompts = []

    negative_prompts = [
        "cartoonish, unrealistic, bad anatomy, worst quality, low quality"
    ]

	

    for img_num in range(NUM_IMAGES_TO_PARSE):
        jpeg_url = (
            f"{IIITD_DATASET_PREFIX}/IIITD-20K/"
            + prompt_data[str(img_num)]["Image ID"]
            + ".jpeg"
        )
        jpg_url = (
            f"{IIITD_DATASET_PREFIX}/IIITD-20K/"
            + prompt_data[str(img_num)]["Image ID"]
            + ".jpg"
        )
        if os.path.exists(jpeg_url):
            image_path = jpeg_url
        elif os.path.exists(jpg_url):
            image_path = jpg_url
        else:
            print("ERROR in file path")
            print(jpeg_url)
            exit(1)

        image = Image.open(image_path)
        image = image.resize((WIDTH//NUM_FRAMES_PER_GEN, HEIGHT))
        image = paste_images_horizontally([image for i in range(NUM_FRAMES_PER_GEN)])
        images.append(image)
        prompts.append(prompt_data[str(img_num)]["Description 1"])

    pipe_I2I, generator = make_pipeline()

    generated_images_per_angle = [
        [] for _ in range(NUM_IMAGES_TO_PARSE)
    ]  # NUM_IMAGES_TO_PARSE * angles

    for img_num in range(NUM_IMAGES_TO_PARSE):
        for angle in range(0, len(cond_pose_images), NUM_FRAMES_PER_GEN):
            control_image = paste_images_horizontally([cond_pose_images[i] for i in range(angle, angle+NUM_FRAMES_PER_GEN)])

            gen_images = pipe_I2I(
                prompt=[prompts[img_num]],
                negative_prompt=negative_prompts,
                # control_image=cond_pose_images[angle],
                control_image=control_image,
                image=generated_images_per_angle[img_num][-1]
                if len(generated_images_per_angle[img_num]) > 0
                else images[img_num],
                generator=generator,
                height=HEIGHT,
                width=WIDTH,
            )

            generated_images_per_angle[img_num].append(gen_images.images[0])

            gen_images = [torch.tensor(np.array(im)) for im in gen_images.images]

            gen_images = [
                torch.permute(im, (2, 0, 1)).unsqueeze(0) for im in gen_images
            ]

            prompt_image = torch.permute(
                torch.tensor(
                    np.array(
                        (
                            generated_images_per_angle[img_num][-2]
                            if len(generated_images_per_angle[img_num]) > 1
                            else images[img_num]
                        ).resize((gen_images[-1].shape[-1], gen_images[-1].shape[-2]))
                    )
                ),
                (2, 0, 1),
            ).unsqueeze(0)


            pose_image = torch.permute(
                torch.tensor(
                    np.array(
                        control_image.resize(
                            (gen_images[-1].shape[-1], gen_images[-1].shape[-2])
                        ).convert("RGB")
                    )
                ),
                (2, 0, 1),
            ).unsqueeze(0)

            gen_images.extend([prompt_image, pose_image])


            grid = torchvision.utils.make_grid(torch.cat(gen_images))
            ndarr = grid
            ndarr = ndarr.to("cpu", torch.uint8).numpy()
            ndarr = ndarr.transpose((1, 2, 0))
            # print(type(ndarr), ndarr.shape)
            im = Image.fromarray(ndarr)

            im.save(SAVE_DIR + f"img_{img_num}_angle_{angle}.png")





if __name__ == "__main__":
    main()
