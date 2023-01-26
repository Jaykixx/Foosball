import os
import numpy as np
from PIL import Image
from environments.Foosball.base import FoosballTask
import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry
import torch


class FoosballCameraTask(FoosballTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        if not hasattr(self, "_num_observations"):
            self._num_observations = 3 * 1280 * 720

        super(FoosballCameraTask, self).__init__(name, sim_config, env, offset)

        # Reset parameters
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_velocity_noise = self._task_cfg["env"]["resetVelocityNoise"]

    def get_camera_sensor(self) -> None:
        rep.WriterRegistry.register(MyWriter)
        camera_path = self.default_zero_env_path + "/Foosball/Top_Down_Camera"
        self.rp = rep.create.render_product(camera_path, resolution=(1280, 720))
        self.writer = rep.WriterRegistry.get("MyWriter")
        self.writer.initialize(rgb=True)
        self.writer.attach([self.rp])
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.rp])
        self.frame_id = 0
        self.frame_path = os.path.join(os.getcwd(), "Foosball_Frames")
        os.makedirs(self.frame_path, exist_ok=True)

    def post_reset(self) -> None:
        super(FoosballCameraTask, self).post_reset()

        if not hasattr(self, 'writer'):
            self.get_camera_sensor()

    def get_observations(self) -> dict:
        rep.orchestrator.step()
        rep.orchestrator.step()
        img = self.rgb.get_data()

        self.obs_buf = img.flatten()

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations


# Access data through a custom replicator writer
class MyWriter(Writer):
    def __init__(self, rgb: bool = True):
        # self.frame_id = 0
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        # Create writer output directory
        # self.file_path = os.path.join(os.getcwd(), "_out_writer", "")
        # dir = os.path.dirname(self.file_path)
        # os.makedirs(dir, exist_ok=True)

    def write(self, data):
        pass
        # for annotator in data.keys():
        #     annotator_split = annotator.split("-")
        #     if len(annotator_split) > 1:
        #         render_product_name = annotator_split[-1]
        #         if annotator.startswith("rgb"):
        #             save_rgb(data[annotator],
        #                      self.file_path + "/" + render_product_name + "_frame_" + str(self.frame_id))
        # self.frame_id += 1


# Save rgb image to file
def save_rgb(rgb_data, file_name):
    rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
    rgb_img = Image.fromarray(rgb_image_data, "RGBA")
    rgb_img.save(file_name + ".png")