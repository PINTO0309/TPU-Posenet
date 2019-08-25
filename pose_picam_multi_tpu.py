import argparse
import numpy as np
import cv2
import time
from PIL import Image
from time import sleep
import multiprocessing as mp
from edgetpu.basic import edgetpu_utils
from pose_engine import PoseEngine
from imutils.video.pivideostream import PiVideoStream
from imutils.video.filevideostream import FileVideoStream
import imutils

lastresults = None
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


def camThread(results, frameBuffer, camera_width, camera_height, model_width, model_height, vidfps, video_file_path):

    global fps
    global detectfps
    global framecount
    global detectframecount
    global time1
    global time2
    global lastresults
    global cam
    global window_name
    global vs

    if video_file_path != "":
        vs = FileVideoStream(video_file_path).start()
        window_name = "Movie File"
    else:
        vs = PiVideoStream((camera_width, camera_height), vidfps).start()
        window_name = "PiCamera"
    time.sleep(2)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        color_image = vs.read()
        if frameBuffer.full():
            frameBuffer.get()
        frames = cv2.resize(color_image, (model_width, model_height))
        frameBuffer.put(frames.copy())
        res = None

        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res, model_width, model_height)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults, model_width, model_height)

        cv2.imshow(window_name, imdraw)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


def inferencer(results, frameBuffer, model, camera_width, camera_height):

    engine = None

    # Acquisition of TPU list without model assignment
    devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)

    devopen = False
    for device in devices:
        try:
            engine = PoseEngine(model, device)
            devopen = True
            break
        except:
            continue

    if devopen == False:
        print("TPU Devices open Error!!!")
        sys.exit(1)

    print("Loaded Graphs!!! ")

    while True:

        if frameBuffer.empty():
            continue

        # Run inference.
        color_image = frameBuffer.get()
        prepimg = color_image[:, :, ::-1].copy()

        tinf = time.perf_counter()
        result, inference_time = engine.DetectPosesInImage(prepimg)
        print(time.perf_counter() - tinf, "sec")
        results.put(result)


def draw_pose(img, pose, threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        img = cv2.circle(img, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5, (0, 255, 0), -1)

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        img = cv2.line(img, (ax, ay), (bx, by), (0, 255, 255), 2)


def overlay_on_image(frames, result, model_width, model_height):

    color_image = frames

    if isinstance(result, type(None)):
        return color_image
    img_cp = color_image.copy()

    for pose in result:
        draw_pose(img_cp, pose)

    cv2.putText(img_cp, fps,       (model_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.putText(img_cp, detectfps, (model_width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

    return img_cp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite", help="Path of the detection model.")
    parser.add_argument('--videofile', default="", help='Path to input video file. (Default="")')
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')
    args = parser.parse_args()

    model           = args.model
    video_file_path = args.videofile
    vidfps          = args.vidfps

    camera_width  = 320
    camera_height = 240
    model_width   = 640
    model_height  = 480

    try:
        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(3)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(results, frameBuffer, camera_width, camera_height, model_width, model_height, vidfps, video_file_path),
                       daemon=True)
        p.start()
        processes.append(p)

        # Activation of inferencer
        devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
        for devnum in range(len(devices)):
            p = mp.Process(target=inferencer,
                           args=(results, frameBuffer, model, model_width, model_height),
                           daemon=True)
            sleep(5)
            p.start()
            processes.append(p)

        while True:
            sleep(1)

    finally:
        for p in range(len(processes)):
            processes[p].terminate()
