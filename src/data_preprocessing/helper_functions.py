import os


def add_leading_zeros(number: str, length=4):
    filled = number.zfill(length)
    return filled


def get_folder_name_from_video_name(video_name):
    video_name = video_name.split('.')[0]  # get stem
    folder_name = add_leading_zeros(video_name)
    return folder_name






