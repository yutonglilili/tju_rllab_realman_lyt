"""
RealSense 单相机 env (ForceFlow 适配版)

接口:
    cam = RealsenseEnv(serial="f1471338", visual_preset="RS2_L500_VISUAL_PRESET_MAX_RANGE")
    obs = cam.step()      # {"rgb": (H,W,3) uint8, "depth": (H,W) uint16, ...meta}
    cam.close()

支持 L515 / D435 等多型号:
- color_resolution / depth_resolution 不传则用相机默认 (因为不同型号支持的分辨率列表不同)
  L515 color 仅支持 1280x720 / 1920x1080 / 960x540
  D435 color 支持 1280x720 / 1920x1080 / 424x240 / 640x480
- visual_preset 是 L515 专有, D435 必须不传 (否则 capture 会返回空帧)
- D435 偶发返回空帧 (open3d t.io 已知行为), 内置 retry + last-good-frame 兜底
"""

import atexit
import time

import numpy as np
import open3d as o3d


class RealsenseEnv:
    def __init__(
        self,
        serial: str,
        color_resolution=None,   # (w, h) 或 None 用相机默认
        depth_resolution=None,
        fps: int = 30,
        visual_preset: str = None,
        record: bool = False,
    ):
        if not serial:
            print(o3d.t.io.RealSenseSensor.list_devices())
            raise ValueError("RealsenseEnv 需要明确的 serial")

        self.serial = serial

        config_dict = {
            "serial": serial,
            "color_format": "RS2_FORMAT_RGB8",
            "depth_format": "RS2_FORMAT_Z16",
            "fps": str(fps),
        }
        if color_resolution is not None:
            config_dict["color_resolution"] = f"{int(color_resolution[0])},{int(color_resolution[1])}"
        if depth_resolution is not None:
            config_dict["depth_resolution"] = f"{int(depth_resolution[0])},{int(depth_resolution[1])}"
        if visual_preset:
            config_dict["visual_preset"] = visual_preset

        self.rs = o3d.t.io.RealSenseSensor()
        config = o3d.t.io.RealSenseSensorConfig(config_dict)
        if record:
            self.rs.init_sensor(config, 0, f"realsense_{serial}.bag")
            self.rs.start_capture(True)
        else:
            self.rs.init_sensor(config, 0)
            self.rs.start_capture()

        md = self.rs.get_metadata()
        self.intrinsic_matrix = md.intrinsics.intrinsic_matrix
        self.depth_scale = md.depth_scale

        self.meta_obs = {
            "serial": serial,
            "size": [md.height, md.width],
            "intrinsic": self.intrinsic_matrix.tolist(),
            "depth_scale": self.depth_scale,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion_model": "none",
        }

        # warmup + 探测实际 shape
        self._last_rgb = None
        self._last_depth = None
        for _ in range(20):
            res = self._capture_once()
            if res is not None:
                self._last_rgb, self._last_depth = res
                break
            time.sleep(0.05)
        if self._last_rgb is None:
            self.close()
            raise RuntimeError(f"RealsenseEnv({serial}) warmup 失败, 没拿到任何有效帧")

        print(f"[RealsenseEnv] {serial}: rgb={self._last_rgb.shape}  depth={self._last_depth.shape}")
        atexit.register(self.close)

    def _capture_once(self):
        """单次 capture, 返回 (rgb_np uint8, depth_np uint16) 或 None"""
        im = self.rs.capture_frame(True, True)
        rgb = np.asarray(im.color)
        depth = np.asarray(im.depth)
        if rgb.size == 0 or depth.size == 0:
            return None
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        return rgb.astype(np.uint8), depth.astype(np.uint16)

    def compute_observation(self, retry_max: int = 5) -> dict:
        for _ in range(retry_max):
            res = self._capture_once()
            if res is not None:
                rgb, depth = res
                self._last_rgb = rgb
                self._last_depth = depth
                return {"rgb": rgb, "depth": depth, **self.meta_obs}
            time.sleep(0.005)

        # retry 用完, 返回上次成功的帧 (warmup 已确保至少有一帧)
        return {"rgb": self._last_rgb, "depth": self._last_depth, **self.meta_obs}

    def reset(self, action=None) -> dict:
        return self.compute_observation()

    def step(self, action=None) -> dict:
        return self.compute_observation()

    def close(self):
        try:
            self.rs.stop_capture()
        except Exception:
            pass


if __name__ == "__main__":
    import cv2

    print(o3d.t.io.RealSenseSensor.list_devices())

    # 双视图预览
    cam_arm = RealsenseEnv(
        serial="342522073663",                     # D435
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
    )
    cam_fix = RealsenseEnv(
        serial="f1471338",                         # L515
        visual_preset="RS2_L500_VISUAL_PRESET_MAX_RANGE",
        # color/depth resolution 留空 -> 用 L515 默认
    )

    try:
        while True:
            obs_arm = cam_arm.step()
            obs_fix = cam_fix.step()

            # resize 到统一尺寸便于拼接展示
            h, w = 360, 480
            arm_view = cv2.cvtColor(cv2.resize(obs_arm["rgb"], (w, h)), cv2.COLOR_RGB2BGR)
            fix_view = cv2.cvtColor(cv2.resize(obs_fix["rgb"], (w, h)), cv2.COLOR_RGB2BGR)
            stacked = np.hstack([arm_view, fix_view])
            cv2.putText(stacked, "arm (D435)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(stacked, "fix (L515)", (w + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("arm | fix", stacked)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam_arm.close()
        cam_fix.close()
        cv2.destroyAllWindows()
