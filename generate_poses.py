from PIL import Image
import numpy as np
import os
import json

from controlnet_aux import OpenposeDetector


dataset_address_prefix = "/raid/home/vibhu20150/Datasets/IIITD-20K/"


with open(dataset_address_prefix + "Filtered.json", "rb") as f:
    prompt_data = json.load(fp=f)

print(prompt_data["0"]["Image ID"])

height = 695
width = 248

# images = []
# prompts = []
# negative_prompts = ["cartoonish, unrealistic, bad anatomy, worst quality, low quality"]
# 
cond_pose_images = []

print(len(prompt_data))


openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
print()

START_IMAGE_ID = 0
END_IMAGE_ID = 20000-1

for i in range(START_IMAGE_ID, END_IMAGE_ID+1):
    print(i)
    image_path = ""

    jpeg_url = (
        f"{dataset_address_prefix}/IIITD-20K/"
        + prompt_data[str(i)]["Image ID"]
        + ".jpeg"
    )
    jpg_url = (
        f"{dataset_address_prefix}/IIITD-20K/"
        + prompt_data[str(i)]["Image ID"]
        + ".jpg"
    )

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

    openpose_image.save(f"/raid/home/vibhu20150/litreview/data/IIITD-20K-openpose/{prompt_data[str(i)]['Image ID']}.png")

    # print(type(openpose_image))

#     images.append(image)
#     cond_pose_images.append(openpose_image)
#     prompts.append(prompt_data[str(i)]["Description 1"])

