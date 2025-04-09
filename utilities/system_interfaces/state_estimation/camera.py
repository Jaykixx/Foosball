import numpy as np
import cv2 as cv
import torch


def filter(img):
    b, g, r = cv.split(img)

    mask_r = r > 130
    mask_g = g < 70
    mask_b = b < 70
    mask = np.minimum(mask_r, np.minimum(mask_b, mask_g))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray[mask] = 255
    gray[~mask] = 0
    return gray


class Camera:

    def __init__(self, config, device='cuda:0'):
        self.device = device
        self.cfg = config

        self.px, self.py = self.cfg['resolution']
        self.fx, self.fy = self.cfg['intrinsic']['focal_length']
        self.lx, self.ly = self.cfg['intrinsic']['sensor_length']
        self.ox, self.oy = int(self.px / 2), int(self.py / 2)

        self.mx = self.px / self.lx
        self.my = self.py / self.ly

        pos = self.cfg.get('position', None)
        if pos is None:
            self.pos = torch.zeros((1, 3), device=self.device)
        else:
            self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
            self.pos = torch.atleast_2d(self.pos)

        rot = self.cfg.get('rotation', None)
        if rot is None:
            rot = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        self.rot = torch.tensor(rot, dtype=torch.float32, device=self.device)
        self.rot = torch.atleast_2d(self.rot)

        # TODO: Fixed by calibration due to inaccurate intrinsic parameters
        self.K = torch.tensor([
            [877.66901, 0, 618.56914],
            [0, 873.03864, 360.40103],
            [0, 0, 1]
        ], device=self.device)
        # self.k = self._compute_k_matrix()  # TODO: Adjust with new camera
        self.p = self._compute_p_matrix()

        # TODO: Specific to windows. CAP_DSHOW does not work on linux
        # Required to push camera to 90 FPS. TODO: Change with new camera
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        self.capture = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv.CAP_PROP_FPS, 90)
        self.capture.set(cv.CAP_PROP_FOURCC, fourcc)

    def _compute_k_matrix(self):
        intrinsics = torch.tensor(
            [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]], device=self.device
        )
        physics = torch.tensor(
            [[self.mx, 0, 0], [0, self.my, 0], [0, 0, 1]], device=self.device
        )
        return intrinsics @ physics

    def _compute_p_matrix(self):
        c = self.rot @ self.pos.T
        transformation = torch.hstack((self.rot, -c))
        return self.k @ transformation

    def calibrate_extrinsics(self):
        # TODO: Currently specific for foosball
        # TODO: Should allow varying methods. Checker vs. Dots
        frame = self.grab_frame()
        gray = filter(frame)
        points_2d = None
        for i in range(20):
            points_2d = cv.HoughCircles(
                gray, cv.HOUGH_GRADIENT, 
                dp=2, minDist=50, param1=50, param2=10, minRadius=5, maxRadius=6
            )
            if points_2d is None or len(points_2d) != 5:
                print("None or not all calibration points found. Retrying...")
            else:
                break
        if points_2d is None:
            raise ValueError('No calibration points found!')
        elif len(points_2d) != 5:
            raise ValueError('Not enough calibration points found!')

        points_2d = points_2d[0, :, :-1].astype(np.float64)
        sorted_2d = points_2d[np.argsort(points_2d[:, 0])]
        sorted_2d[:2] = sorted_2d[:2][np.argsort(sorted_2d[:2, 1])]
        sorted_2d[3:] = sorted_2d[3:][np.argsort(sorted_2d[3:, 1])]
        
        x, y = 0.315, 0.21
        points_3d = np.array(
            [[-x, y, 0], [-x, -y, 0], [0, 0, 0], [x, y, 0], [x, -y, 0]],
            dtype=np.float64
        )

        dist_coeffs = np.zeros((4, 1))
        _, rvec, tvec = cv.solvePnP(
            points_3d, sorted_2d, self.k.cpu().numpy(), dist_coeffs
        )
        
        rot, _ = cv.Rodrigues(rvec)
        pos = -np.linalg.inv(rot) @ tvec
        print(f"Camera Position: {pos}")
        self.pos = torch.from_numpy(pos.astype(np.float32)).to(self.device)
        self.rot = torch.from_numpy(rot.astype(np.float32)).to(self.device)
        self.k = self._compute_k_matrix()
        self.p = self._compute_p_matrix()

    def grab_frame(self):
        ret = False
        while not ret:
            ret, frame = self.capture.read()
            if not ret:
                raise ConnectionError("Camera connection was not properly established!")
        return frame

    def _remove_distortion(self):
        pass

    def projection(self, xyz):
        """ Projects points from world coordinates into the image frame """
        # TODO: Implement
        pass

    def inverse_projection(self, xy_img, z_wcs):
        """ Projects points from the image frame into world coordinates """
        # TODO: Validate
        # put the 2d images point in a homogenous vector im image coordinate
        xy_h = torch.hstack(
            [xy_img, torch.ones((xy_img.shape[0], 1), device=self.device)]
        )

        # calculate inverse KR matrix
        krm = torch.inverse(self.k @ self.rot).T

        # calculate lambda
        lmbda = (z_wcs - self.pos[0, 2]) / (xy_h @ krm)

        # compute reprojected points in the world coordinate system
        xy = (self.pos + lmbda * (xy_h @ krm))[..., :2]
        return xy
