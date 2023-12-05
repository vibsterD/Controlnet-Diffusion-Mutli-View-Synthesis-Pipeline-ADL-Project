import cv2
import os

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
    video_paths = ["./vibhu.mp4", "./niranjan.mp4" ]
    output_folder = ["./output_frames/vibhu", "./output_frames/niranjan"]

    


if __name__ == "__main__":
    video_path = "./vibhu.mp4" 
    output_folder = "./output_frames/vibhu"

    generate_frames(video_path, output_folder)
