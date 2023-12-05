import cv2
import os


from controlnet_aux import OpenposeDetector
from PIL import Image

from tqdm import tqdm

def generate_frames(video_path, output_folder):

    video_capture = cv2.VideoCapture(video_path)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_number in range(frame_count):
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_filename = f"{output_folder}/frame_{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)

    video_capture.release()


def generate_dataset():

    HEIGHT = 693
    WIDTH = 249

    video_paths = ["./vibhu.mp4", "./niranjan.mp4" ]
    output_folders = ["./output_frames/vibhu", "./output_frames/niranjan"]

    for i in range(len(video_paths)):
        

        generate_frames(video_paths[i], output_folders[i])


    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    for i in range(len(video_paths)):
        out_pose_folder = f"./poses/{video_paths[i].split('/')[-1]}/"

        if not os.path.exists(out_pose_folder):
            os.makedirs(out_pose_folder)

        image_paths = os.listdir(output_folders[i])
        image_paths.sort()
        image_paths = [output_folders[i] +  "/" + im_path for im_path in image_paths]

        for im_path in tqdm(image_paths):
            image = Image.open(im_path)
            image = image.resize((WIDTH, HEIGHT))

            pose_img = openpose(image).resize((WIDTH, HEIGHT))

            pose_img.save(out_pose_folder + im_path.split('/')[-1])




    

    


if __name__ == "__main__":
    # video_path = "./vibhu.mp4" 
    # output_folder = "./output_frames/vibhu"

    # generate_frames(video_path, output_folder)

    generate_dataset()
