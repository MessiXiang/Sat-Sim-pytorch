import torch
from satsim.architecture import Module


class Camera(Module):

    def __init__(self, *args, **kwargs) -> None:
        self.filename = ''
        self.save_dir = ''
        self.sensor_time_tag = 0
        self.save_images = 0

        # Camera parameters
        self.parent_name = ''
        self.camera_is_on = 0
        self.camera_id = 1
        self.resolution = (512, 512)
        self.render_rate = 60 * 1e9
        self.field_of_view = 0.7
        self.camera_pos_b = torch.zeros(3)
        self.sigma_cb = torch.zeros(3)
        self.sky_box = "black"
        self.post_processing_on = 0
        self.pp_focus_distance = 0.
        self.pp_aperture = 0.
        self.pp_focal_length = 0.
        self.pp_max_blur_size = 0

        # Noise parameters
        self.gaussian = 0.
        self.dark_curernt = 0.
        self.salt_pepper = 0.
        self.cosmic_ray = 0.
        self.blur_param = 0.
        self.hsv = torch.zeros(3)
        self.bgr_percent = torch.zeros(3)

    def update_state(self):
        pass

    def reset(self):
        pass

    def hsv_adjust(self, ):
        pass

    def bgr_adjust_percent(self):
        pass

    def add_gaussian_noise(self):
        pass

    def add_sault_pepper(self):
        pass

    def add_cosmic_ray(self):
        pass

    def add_cosmic_ray_burst(self):
        pass

    def apply_filters(self):
        pass
