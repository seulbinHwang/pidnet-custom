import os
import cv2


def split_video_to_frames(video_path, output_folder, filename,
                          generated_frame_count):
    # 동영상을 열고 총 프레임 수를 확인합니다.
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 프레임을 순회하면서 JPG 이미지로 저장합니다.
    for frame_index in range(total_frames):
        if frame_index % 30 != 0:
            continue
        # 현재 프레임을 읽어옵니다.
        ret, frame = video_capture.read()

        if ret:
            # 프레임을 JPG 이미지로 변환하여 저장합니다.
            # I want to save file name as combination of filename and frame_index.
            file_name = f'{generated_frame_count}_{filename}.jpg'
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, frame)
            generated_frame_count += 1
        else:
            break
    print(f'Generated {generated_frame_count} frames.')
    # 사용한 자원을 해제합니다.
    video_capture.release()
    return generated_frame_count


# 동영상이 저장된 폴더 경로와 프레임 이미지를 저장할 폴더 경로를 지정합니다.
video_folder = './samples/video_data'
output_folder = './samples/raw_data_from_video'

import cv2

# 폴더 내의 모든 MP4 파일에 대해 분할 작업을 수행합니다.
generated_frame_count = 0

for filename in os.listdir(video_folder):
    print("filename:", filename)

    if filename.endswith('.mp4') or filename.endswith('.MOV'):
        video_path = os.path.join(video_folder, filename)
        generated_frame_count = split_video_to_frames(video_path, output_folder,
                                                      filename,
                                                      generated_frame_count)
