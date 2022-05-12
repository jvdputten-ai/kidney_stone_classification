from src.data_preprocessing.helper_functions import get_folder_name_from_video_name
import numpy as np
import cv2
import os


class FrameExtractor:
    def __init__(self, root, vid_source, frame_destination):
        self.root = root
        self.video_folder = root / vid_source
        self.frame_folder = root / frame_destination
        self.video_list = self.get_video_list()
        self.create_folder_structure_from_videos()

    def get_video_list(self):
        video_list = [item for item in os.listdir(self.video_folder) if
                      os.path.isfile(self.video_folder / item)]
        return video_list

    def create_folder_structure_from_videos(self):
        for video_name in self.video_list:
            folder_name = get_folder_name_from_video_name(video_name)
            if not os.path.isdir(self.frame_folder / folder_name):
                os.mkdir(self.frame_folder / folder_name)

    def process_videos(self):
        for vid in self.video_list:
            print(f'processing video {vid}')
            folder_name = get_folder_name_from_video_name(vid)
            frame_destination_folder = self.frame_folder / folder_name
            self.extract_unique_frames(frame_destination_folder, vid)

    def extract_unique_frames(self, frame_destination_folder, vid):
        cap = cv2.VideoCapture(str(self.video_folder / vid))
        min_frame_distance: int = cap.get(cv2.CAP_PROP_FPS) // 1
        count = 1
        old_count = -1000
        new_frame = True
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if count == 1:
                border = self.get_border(frame)
            cropped_frame = self.crop_border(frame, border)

            if new_frame and ( (count - old_count) > min_frame_distance):
                frame_name = str(count).zfill(4) + '.png'
                resized_frame = cv2.resize(cropped_frame, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(frame_destination_folder / frame_name), resized_frame)
                old_count = count
                prev_frame = cropped_frame
                new_frame = False

            else:
                if not self.frames_are_equal(cropped_frame, prev_frame):
                    prev_frame = cropped_frame
                    new_frame = True

            count += 1

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def frames_are_equal(frame1, frame2):
        h, w, _ = frame1.shape
        return not (np.bitwise_xor(frame1[h//2-10:h//2+10, w//2-10:w//2+10, :],
                                   frame2[h//2-10:h//2+10, w//2-10:w//2+10, :]).any())


    @staticmethod
    def get_border(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        return x, y, w, h

    @staticmethod
    def crop_border(img, border):
        (x, y, w, h) = border
        crop = img[y:y + h, x:x + w]
        return crop



