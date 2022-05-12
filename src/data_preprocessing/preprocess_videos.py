from pathlib import Path
from src.data_preprocessing.frame_extractor import FrameExtractor

DATA_ROOT = Path('D:\\Research projects\\stoneclassification\\dataset')
SOURCE_FOLDER = 'videos'
DEST_FOLDER = 'frames_preprocessed'

frame_extractor = FrameExtractor(DATA_ROOT, SOURCE_FOLDER, DEST_FOLDER)
frame_extractor.process_videos()


