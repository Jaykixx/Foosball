from utilities.system_interfaces.state_estimation.camera import Camera
import numpy as np
import cv2 as cv
import torch
import time
import os


# TODO: Introduce Builder for different models and keep this as wrapper
class ObjectDetector:

    def __init__(self, cfg, device='cuda:0'):
        self.cfg = cfg
        self.device = device

        self.model_name = 'ultralytics/' + self.cfg['model']
        self.object_classes = self.cfg['object_classes']
        self.objects = self.cfg['objects']

        checkpoint = self.cfg['checkpoint']
        dir_path = os.getcwd()
        self.checkpoint = os.path.join(dir_path, checkpoint)

        self.model = torch.hub.load(
            self.model_name, 'custom', path=self.checkpoint
        )

        self.camera = Camera(self.cfg['camera'], self.device)
        self.save_results = self.cfg['save_results']
        if self.save_results:
            self.save_path = os.path.join(dir_path, self.cfg['save_path'])
            self.frames = []
            self.detection_results = []

    def get_observation(self):
        clk = time.perf_counter()
        frame = self.camera.grab_frame()
        # frame = cv.addWeighted(frame, 1, np.zeros(frame.shape, frame.dtype), 0, 10)
        frame_time = time.perf_counter() - clk

        clk = time.perf_counter()
        # Switch to RGB and run detection
        with torch.no_grad():
            result = self.model(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        pos = result.xywh[0]
        # cv.imshow("Stream", cv.cvtColor(result.render()[0], cv.COLOR_RGB2BGR))
        # cv.waitKey(1)
        positions = {}
        for k in self.object_classes:
            if k == 1:  # Get only best ball detection
                p = pos[pos[:, -1] == k]
                p = torch.max(p, dim=-2, keepdim=True)[0][:, :2]
            else:
                p = pos[pos[:, -1] == k][:, :2]
            name = self.objects[k]['name']
            if p.numel() > 0:
                positions[name] = self.camera.inverse_projection(p, self.objects[k]['height'])
            else:
                positions[name] = None
        inference_time = time.perf_counter() - clk

        if self.save_results:
            self.frames.append(frame)
            self.detection_results.append(result)

        times = {
            'Frame Time': frame_time,
            'Frame Inference Time': inference_time
        }

        result = {
            'positions': positions,
            'detection_result': result,
            'frame': frame,
            'times': times
        }

        return result
