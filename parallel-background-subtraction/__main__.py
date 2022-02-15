from mpi4py import MPI
import cv2 as cv
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

INPUT_FILE = 'input/vtest.avi'
OUTPUT_FOLDER = 'output/'
OUTPUT_VIDEO = OUTPUT_FOLDER + 'output_vid.avi'
NUM_HEIGHT_SLICES = 2
NUM_WIDTH_SLICES = size // NUM_HEIGHT_SLICES

assert size % 2 == 0

if rank == 0:
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(INPUT_FILE))
    _, frame = capture.read()

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    fps = capture.get(cv.CAP_PROP_FPS)
    video_writer = cv.VideoWriter(
        OUTPUT_VIDEO, fourcc, fps, frame.shape[1::-1], True)
else:
    capture = None

backSub = cv.createBackgroundSubtractorKNN()
frameCount = 0

while True:
    height_slice = None
    width_slice = None
    frame = None
    frame_chunks = None

    if rank == 0:
        _, frame = capture.read()
        frameCount += 1
        if frame is None:
            frame_chunks = [None] * size
        else:
            frame_chunks = []
            height_slice = frame.shape[1] // NUM_HEIGHT_SLICES
            width_slice = frame.shape[0] // NUM_WIDTH_SLICES
            for i in range(NUM_WIDTH_SLICES):
                for j in range(2):
                    frame_chunks.append(frame[i * width_slice:(
                        (i + 1) * width_slice), j * height_slice:((j + 1) * height_slice)])

    chunk = comm.scatter(frame_chunks, root=0)
    if chunk is None:
        break

    foregroundMask = backSub.apply(chunk)
    median = cv.medianBlur(foregroundMask, 3)

    maskedChunks = comm.gather(median, root=0)

    if rank == 0:
        new_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for i in range(NUM_WIDTH_SLICES):
            for j in range(2):
                new_mask[i * width_slice:(
                    (i + 1) * width_slice), j * height_slice:((j + 1) * height_slice)] = maskedChunks[i * 2 + j]

        merged_frame = cv.merge(
            [new_mask, new_mask, new_mask])  # grayscale to rgb

        video_writer.write(merged_frame)
