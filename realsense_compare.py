import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app_logging import get_logger
from calibrate_d_to_r import calibration_values, save_aligned_point_cloud

from threading import RLock


LOGGER = get_logger("realsense")


def _write_image(path, image):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Image encoding failed: {path}")
    path.write_bytes(encoded.tobytes())


def _rs():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError as error:
        raise RuntimeError("pyrealsense2가 필요합니다: pip install pyrealsense2") from error


def _profile(device, stream, index, width, height, fmt=None):
    rs = _rs()
    for sensor in device.sensors:
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            if (profile.stream_type() == stream and profile.stream_index() == index and
                    video.width() == width and video.height() == height and
                    (fmt is None or profile.format() == fmt)):
                return profile
    raise RuntimeError(f"RealSense profile not found: {stream} {width}x{height}")


def _intrinsic(profile):
    value = profile.as_video_stream_profile().get_intrinsics()
    return {"K": [[value.fx, 0, value.ppx], [0, value.fy, value.ppy], [0, 0, 1]],
            "dist": list(value.coeffs), "model": str(value.model)}


def save_factory_calibration(path, target_kind, width, height):
    rs = _rs()
    devices = list(rs.context().query_devices())
    if not devices:
        raise RuntimeError("연결된 RealSense 카메라가 없습니다.")
    device = devices[0]
    depth = _profile(device, rs.stream.depth, 0, width, height, rs.format.z16)
    stream, index, fmt = ((rs.stream.infrared, 2, rs.format.y8) if target_kind == "right_ir"
                          else (rs.stream.color, 0, rs.format.rgb8))
    target = _profile(device, stream, index, width, height, fmt)
    di, ti = _intrinsic(depth), _intrinsic(target)
    extrinsic = depth.get_extrinsics_to(target)
    result = {"source": "factory", "device": device.get_info(rs.camera_info.name),
              "serial": device.get_info(rs.camera_info.serial_number),
              "target_stream": target_kind, "width": width, "height": height,
              "depth_K": di["K"], "depth_dist": di["dist"], "depth_model": di["model"],
              "target_K": ti["K"], "target_dist": ti["dist"], "target_model": ti["model"],
              "depth_scale": device.first_depth_sensor().get_depth_scale(),
              "R": np.asarray(extrinsic.rotation).reshape(3, 3).T.tolist(),
              "t": list(extrinsic.translation)}
    result["RT"] = np.block([[np.asarray(result["R"]), np.asarray(result["t"])[:, None]],
                              [np.zeros((1, 3)), np.ones((1, 1))]]).tolist()
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _intrinsics(rs, K, dist, width, height, model=""):
    value = rs.intrinsics()
    value.width, value.height = width, height
    value.fx, value.fy, value.ppx, value.ppy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    value.coeffs = np.pad(np.asarray(dist).ravel(), (0, max(0, 5 - np.size(dist))))[:5].tolist()
    value.model = (rs.distortion.inverse_brown_conrady if "inverse" in model.lower()
                   else rs.distortion.brown_conrady)
    return value


# def rs_align_depth(frame, calibration):
#     started = perf_counter()
#     rs = _rs()
#     Kd, dd, Kt, dt, scale = calibration_values(frame, calibration)
#     height, width = frame["right"].shape[:2]
#     target_kind = calibration.get("target_stream", "color" if frame["right"].ndim == 3 else "right_ir")
#     device = rs.software_device()
#     depth_sensor, target_sensor = device.add_sensor("Depth"), device.add_sensor("Target")

#     depth_stream = rs.video_stream()
#     depth_stream.type, depth_stream.index, depth_stream.uid = rs.stream.depth, 0, 1
#     depth_stream.width, depth_stream.height, depth_stream.fps = frame["depth"].shape[1], frame["depth"].shape[0], 30
#     depth_stream.bpp, depth_stream.fmt = 2, rs.format.z16
#     depth_stream.intrinsics = _intrinsics(rs, Kd, dd, depth_stream.width, depth_stream.height,
#                                           calibration.get("depth_model", ""))
#     depth_profile = depth_sensor.add_video_stream(depth_stream)

#     image = frame["right"]
#     if target_kind == "right_ir" and image.ndim == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif target_kind == "color" and image.ndim == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     target_stream = rs.video_stream()
#     target_stream.type = rs.stream.infrared if target_kind == "right_ir" else rs.stream.color
#     target_stream.index, target_stream.uid = (2 if target_kind == "right_ir" else 0), 2
#     target_stream.width, target_stream.height, target_stream.fps = width, height, 30
#     target_stream.bpp = 1 if target_kind == "right_ir" else 3
#     target_stream.fmt = rs.format.y8 if target_kind == "right_ir" else rs.format.bgr8
#     target_stream.intrinsics = _intrinsics(rs, Kt, dt, width, height,
#                                            calibration.get("target_model", ""))
#     target_profile = target_sensor.add_video_stream(target_stream)

#     extrinsic = rs.extrinsics()
#     extrinsic.rotation = np.asarray(calibration["R"]).T.ravel().tolist()
#     extrinsic.translation = list(calibration["t"])
#     depth_profile.register_extrinsics_to(target_profile, extrinsic)
#     device.create_matcher(rs.matchers.default)
#     syncer = rs.syncer()
#     depth_sensor.open(depth_profile)
#     target_sensor.open(target_profile)
#     depth_sensor.start(syncer)
#     target_sensor.start(syncer)
#     LOGGER.info("rs.align start size=%dx%d target=%s", width, height, target_kind)
#     try:
#         streams = ((depth_sensor, np.ascontiguousarray(frame["depth"], np.uint16), depth_profile, 2, scale),
#                    (target_sensor, np.ascontiguousarray(image), target_profile, target_stream.bpp, 0))
#         for number in range(3):
#             for sensor, pixels, profile, bpp, units in streams:
#                 video = rs.software_video_frame()
#                 video.pixels, video.stride, video.bpp = pixels, pixels.shape[1] * bpp, bpp
#                 video.timestamp, video.frame_number = number * 1000 / 30, number
#                 video.domain, video.profile = rs.timestamp_domain.hardware_clock, profile.as_video_stream_profile()
#                 if units:
#                     video.depth_units = units
#                 sensor.on_video_frame(video)
#         LOGGER.info("rs.align waiting for software frames")
#         frames = syncer.wait_for_frames(5000)
#         LOGGER.info("rs.align processing frames")
#         processed = rs.align(target_stream.type).process(frames)
#         aligned = processed.get_depth_frame()
#         if not aligned:
#             streams = [(str(f.profile.stream_type()), f.profile.stream_index()) for f in processed]
#             raise RuntimeError(f"rs.align returned no depth frame: {streams}")
#         depth = np.asanyarray(aligned.get_data()).copy().astype(np.float32) * scale
#         depth[(depth < frame.get("z_min", .15)) | (depth > frame.get("z_max", 2.0))] = 0
#         LOGGER.info("rs.align done elapsed=%.3fs valid=%d", perf_counter() - started,
#                     np.count_nonzero(depth))
#         return depth
#     finally:
#         depth_sensor.stop()
#         target_sensor.stop()
#         depth_sensor.close()
#         target_sensor.close()
#         LOGGER.info("rs.align software streams closed")


# ----------------------------------------------------------------------
# rs.align용 RealSense software_device 세션 캐시
#
# 기존 방식:
#   rs_align_depth() 호출할 때마다
#   software_device / sensor / syncer 생성
#   → 계산
#   → stop / close
#
# 변경 방식:
#   같은 카메라 설정 / RT에서는 세션을 한 번만 만들고 계속 재사용한다.
#   이미지가 바뀌어도 frame 데이터만 새로 주입한다.
# ----------------------------------------------------------------------

_RS_ALIGN_LOCK = RLock()
_RS_ALIGN_SESSIONS = {}


def _rs_align_array_key(value):
    """numpy 배열을 세션 설정 비교용 key로 변환한다."""
    array = np.ascontiguousarray(value, dtype=np.float64)
    return array.shape, array.tobytes()


def _rs_align_session_key(
    frame,
    calibration,
    Kd,
    dd,
    Kt,
    dt,
    scale,
    target_kind,
):
    """
    rs.align 세션을 새로 만들어야 하는 설정인지 판단하기 위한 key.

    이미지 내용은 key에 포함하지 않는다.
    카메라 파라미터 / 해상도 / RT 등이 같으면 같은 세션을 사용한다.
    """
    return (
        tuple(frame["depth"].shape[:2]),
        tuple(frame["right"].shape[:2]),
        target_kind,

        str(calibration.get("depth_model", "")),
        str(calibration.get("target_model", "")),

        float(scale),

        _rs_align_array_key(Kd),
        _rs_align_array_key(dd),
        _rs_align_array_key(Kt),
        _rs_align_array_key(dt),

        _rs_align_array_key(calibration["R"]),
        _rs_align_array_key(calibration["t"]),
    )


class _RsAlignSession:
    """
    RealSense software_device와 sensor를 계속 유지하면서
    여러 이미지의 Depth를 rs.align()하는 세션.
    """

    def __init__(
        self,
        rs,
        frame,
        calibration,
        Kd,
        dd,
        Kt,
        dt,
        scale,
        target_kind,
    ):
        self.rs = rs
        self.scale = float(scale)
        self.target_kind = target_kind
        self.closed = False

        self.depth_height, self.depth_width = frame["depth"].shape[:2]
        self.target_height, self.target_width = frame["right"].shape[:2]

        # --------------------------------------------------------------
        # 1. software_device 생성
        #    중요: 이미지마다 다시 만들지 않고 세션 동안 계속 유지
        # --------------------------------------------------------------
        self.device = rs.software_device()

        self.depth_sensor = self.device.add_sensor("Depth")
        self.target_sensor = self.device.add_sensor("Target")

        # --------------------------------------------------------------
        # 2. Depth stream 생성
        # --------------------------------------------------------------
        depth_stream = rs.video_stream()

        depth_stream.type = rs.stream.depth
        depth_stream.index = 0
        depth_stream.uid = 1

        depth_stream.width = self.depth_width
        depth_stream.height = self.depth_height
        depth_stream.fps = 30

        depth_stream.bpp = 2
        depth_stream.fmt = rs.format.z16

        depth_stream.intrinsics = _intrinsics(
            rs,
            Kd,
            dd,
            self.depth_width,
            self.depth_height,
            calibration.get("depth_model", ""),
        )

        self.depth_profile = self.depth_sensor.add_video_stream(
            depth_stream
        )

        # --------------------------------------------------------------
        # 3. Target stream 생성
        # --------------------------------------------------------------
        target_stream = rs.video_stream()

        if target_kind == "right_ir":
            target_stream.type = rs.stream.infrared
            target_stream.index = 2
            target_stream.bpp = 1
            target_stream.fmt = rs.format.y8

            self.target_bpp = 1

        else:
            target_stream.type = rs.stream.color
            target_stream.index = 0
            target_stream.bpp = 3
            target_stream.fmt = rs.format.bgr8

            self.target_bpp = 3

        target_stream.uid = 2

        target_stream.width = self.target_width
        target_stream.height = self.target_height
        target_stream.fps = 30

        target_stream.intrinsics = _intrinsics(
            rs,
            Kt,
            dt,
            self.target_width,
            self.target_height,
            calibration.get("target_model", ""),
        )

        self.target_stream_type = target_stream.type

        self.target_profile = self.target_sensor.add_video_stream(
            target_stream
        )

        # --------------------------------------------------------------
        # 4. Depth → Target extrinsic 등록
        # --------------------------------------------------------------
        extrinsic = rs.extrinsics()

        extrinsic.rotation = (
            np.asarray(calibration["R"])
            .T
            .ravel()
            .tolist()
        )

        extrinsic.translation = list(
            np.asarray(calibration["t"]).ravel()
        )

        self.depth_profile.register_extrinsics_to(
            self.target_profile,
            extrinsic,
        )

        # --------------------------------------------------------------
        # 5. matcher / syncer / align 생성
        #
        # 이것들도 매 이미지마다 만들지 않는다.
        # --------------------------------------------------------------
        self.device.create_matcher(
            rs.matchers.default
        )

        self.syncer = rs.syncer()

        self.aligner = rs.align(
            self.target_stream_type
        )

        # --------------------------------------------------------------
        # 6. sensor open/start
        #
        # 여기서 딱 한 번 시작한다.
        # --------------------------------------------------------------
        self.depth_sensor.open(
            self.depth_profile
        )

        self.target_sensor.open(
            self.target_profile
        )

        self.depth_sensor.start(
            self.syncer
        )

        self.target_sensor.start(
            self.syncer
        )

        # software frame 번호
        # 이미지가 바뀌어도 계속 증가시킨다.
        self.frame_number = 0

        LOGGER.info(
            "rs.align session created "
            "depth=%dx%d target=%dx%d kind=%s",
            self.depth_width,
            self.depth_height,
            self.target_width,
            self.target_height,
            self.target_kind,
        )

    def process(self, frame):
        """
        이미 생성된 software_device / sensor / syncer를 재사용하면서
        새로운 이미지의 Depth + Target frame을 주입하고 rs.align 결과를 반환한다.

        syncer는 초기 몇 frame에서 Depth와 Target을 동시에 포함하지 않은
        불완전한 frameset을 반환할 수 있으므로, 완전한 frameset이 나올 때까지
        최대 5회 frame pair를 주입한다.
        """
        if self.closed:
            raise RuntimeError(
                "이미 종료된 rs.align session입니다."
            )

        # --------------------------------------------------------------
        # Target 영상 준비
        # --------------------------------------------------------------
        image = frame["right"]

        if self.target_kind == "right_ir":
            if image.ndim == 3:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

            target_pixels = np.ascontiguousarray(
                image,
                dtype=np.uint8,
            )

        else:
            if image.ndim == 2:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR,
                )

            target_pixels = np.ascontiguousarray(
                image,
                dtype=np.uint8,
            )

        depth_pixels = np.ascontiguousarray(
            frame["depth"],
            dtype=np.uint16,
        )

        # --------------------------------------------------------------
        # syncer가 완전한 Depth + Target frameset을 만들 때까지
        # 최대 몇 번 반복해서 같은 이미지를 주입한다.
        #
        # 실제 이미지 내용은 같지만 frame_number와 timestamp는 계속 증가한다.
        # --------------------------------------------------------------
        frames = None

        max_attempts = 5

        for attempt in range(max_attempts):

            number = self.frame_number
            self.frame_number += 1

            timestamp = number * 1000.0 / 30.0

            LOGGER.info(
                "rs.align feed frame=%d attempt=%d/%d target=%s",
                number,
                attempt + 1,
                max_attempts,
                self.target_kind,
            )

            # ----------------------------------------------------------
            # Depth software frame
            # ----------------------------------------------------------
            depth_video = self.rs.software_video_frame()

            depth_video.pixels = depth_pixels
            depth_video.stride = self.depth_width * 2
            depth_video.bpp = 2

            depth_video.timestamp = timestamp
            depth_video.frame_number = number

            depth_video.domain = (
                self.rs.timestamp_domain.hardware_clock
            )

            depth_video.profile = (
                self.depth_profile.as_video_stream_profile()
            )

            depth_video.depth_units = self.scale

            self.depth_sensor.on_video_frame(
                depth_video
            )

            # ----------------------------------------------------------
            # Target software frame
            # ----------------------------------------------------------
            target_video = self.rs.software_video_frame()

            target_video.pixels = target_pixels
            target_video.stride = (
                self.target_width * self.target_bpp
            )
            target_video.bpp = self.target_bpp

            target_video.timestamp = timestamp
            target_video.frame_number = number

            target_video.domain = (
                self.rs.timestamp_domain.hardware_clock
            )

            target_video.profile = (
                self.target_profile.as_video_stream_profile()
            )

            self.target_sensor.on_video_frame(
                target_video
            )

            # ----------------------------------------------------------
            # sync된 frameset 대기
            # ----------------------------------------------------------
            LOGGER.info(
                "rs.align waiting for software frames frame=%d",
                number,
            )

            candidate = self.syncer.wait_for_frames(
                5000
            )

            # ----------------------------------------------------------
            # candidate 안에 실제로 Depth와 Target이 둘 다 있는지 검사
            # ----------------------------------------------------------
            has_depth = False
            has_target = False

            stream_list = []

            for f in candidate:
                profile = f.profile

                stream_type = profile.stream_type()
                stream_index = profile.stream_index()

                stream_list.append(
                    (
                        str(stream_type),
                        stream_index,
                    )
                )

                if stream_type == self.rs.stream.depth:
                    has_depth = True

                if self.target_kind == "right_ir":
                    if (
                        stream_type == self.rs.stream.infrared
                        and stream_index == 2
                    ):
                        has_target = True

                else:
                    if stream_type == self.rs.stream.color:
                        has_target = True

            LOGGER.info(
                "rs.align synced frame=%d streams=%s "
                "depth=%s target=%s",
                number,
                stream_list,
                has_depth,
                has_target,
            )

            # Depth + Target 둘 다 존재할 때만 rs.align()에 넘긴다.
            if has_depth and has_target:
                frames = candidate
                break

            LOGGER.warning(
                "rs.align incomplete frameset "
                "frame=%d attempt=%d/%d streams=%s",
                number,
                attempt + 1,
                max_attempts,
                stream_list,
            )

        # --------------------------------------------------------------
        # 5회 시도해도 Depth + Target을 같이 받지 못했다면 실패
        # --------------------------------------------------------------
        if frames is None:
            raise RuntimeError(
                "rs.align sync failed: "
                f"Depth + {self.target_kind} frameset을 "
                f"{max_attempts}회 시도해도 만들지 못했습니다."
            )

        # --------------------------------------------------------------
        # 완전한 frameset에 대해서만 rs.align 실행
        # --------------------------------------------------------------
        LOGGER.info(
            "rs.align processing complete frameset"
        )

        processed = self.aligner.process(
            frames
        )

        aligned = processed.get_depth_frame()

        if not aligned:
            streams = [
                (
                    str(f.profile.stream_type()),
                    f.profile.stream_index(),
                )
                for f in processed
            ]

            raise RuntimeError(
                "rs.align returned no depth frame: "
                f"{streams}"
            )

        # --------------------------------------------------------------
        # aligned Depth -> numpy [m]
        # --------------------------------------------------------------
        depth = (
            np.asanyarray(aligned.get_data())
            .copy()
            .astype(np.float32)
            * self.scale
        )

        # 기존 코드와 동일하게 Z 범위 적용
        depth[
            (depth < frame.get("z_min", .15))
            |
            (depth > frame.get("z_max", 2.0))
        ] = 0

        return depth

    def close(self):
        """
        세션 종료.
        프로그램 실행 중에는 일반적으로 호출하지 않고,
        프로그램 자체를 종료할 때 한 번 호출한다.
        """
        if self.closed:
            return

        self.closed = True

        LOGGER.info(
            "rs.align session closing target=%s",
            self.target_kind,
        )

        try:
            self.depth_sensor.stop()
        except Exception:
            LOGGER.exception(
                "rs.align depth sensor stop failed"
            )

        try:
            self.target_sensor.stop()
        except Exception:
            LOGGER.exception(
                "rs.align target sensor stop failed"
            )

        try:
            self.depth_sensor.close()
        except Exception:
            LOGGER.exception(
                "rs.align depth sensor close failed"
            )

        try:
            self.target_sensor.close()
        except Exception:
            LOGGER.exception(
                "rs.align target sensor close failed"
            )

        LOGGER.info(
            "rs.align session closed target=%s",
            self.target_kind,
        )


def rs_align_depth(frame, calibration):
    """
    RealSense rs.align() 결과를 계산한다.

    같은 카메라 설정 / RT에서는 software_device와 sensor를
    새로 만들지 않고 기존 세션을 계속 재사용한다.
    """
    started = perf_counter()

    rs = _rs()

    Kd, dd, Kt, dt, scale = calibration_values(
        frame,
        calibration,
    )

    target_kind = calibration.get(
        "target_stream",
        "color"
        if frame["right"].ndim == 3
        else "right_ir",
    )

    key = _rs_align_session_key(
        frame,
        calibration,
        Kd,
        dd,
        Kt,
        dt,
        scale,
        target_kind,
    )

    # RealSense 세션은 동시에 두 곳에서 사용하지 않도록 보호한다.
    with _RS_ALIGN_LOCK:
        session = _RS_ALIGN_SESSIONS.get(key)

        if session is None:
            LOGGER.info(
                "rs.align session cache MISS "
                "target=%s cached_sessions=%d",
                target_kind,
                len(_RS_ALIGN_SESSIONS),
            )

            session = _RsAlignSession(
                rs,
                frame,
                calibration,
                Kd,
                dd,
                Kt,
                dt,
                scale,
                target_kind,
            )

            _RS_ALIGN_SESSIONS[key] = session

        else:
            LOGGER.info(
                "rs.align session cache HIT "
                "target=%s frame=%d",
                target_kind,
                session.frame_number,
            )

        try:
            depth = session.process(
                frame
            )

        except Exception:
            # 정상적인 실행에서는 들어오지 않아야 한다.
            #
            # 세션 자체가 깨졌다면 다음 호출에서는 새로 만들 수 있도록
            # cache에서 제거하고 해당 세션만 종료한다.
            LOGGER.exception(
                "rs.align session process failed; "
                "discarding session"
            )

            _RS_ALIGN_SESSIONS.pop(
                key,
                None,
            )

            session.close()

            raise

    LOGGER.info(
        "rs.align done elapsed=%.3fs valid=%d",
        perf_counter() - started,
        np.count_nonzero(depth),
    )

    return depth


def close_rs_align_sessions():
    """
    프로그램을 완전히 종료할 때만 호출한다.

    실행 중 이미지가 바뀌거나 체크박스를 껐다고
    이 함수를 호출하면 안 된다.
    """
    with _RS_ALIGN_LOCK:
        sessions = list(
            _RS_ALIGN_SESSIONS.values()
        )

        _RS_ALIGN_SESSIONS.clear()

    LOGGER.info(
        "rs.align closing all sessions count=%d",
        len(sessions),
    )

    for session in sessions:
        session.close()






def comparison_metrics(custom, realsense):
    a, b = custom > 0, realsense > 0
    common = a & b
    error = (custom[common] - realsense[common]) * 1000
    return {"custom_valid_percent": float(100 * a.mean()),
            "rs_align_valid_percent": float(100 * b.mean()),
            "overlap_percent": float(100 * common.mean()),
            "overlap_iou_percent": float(100 * common.sum() / max((a | b).sum(), 1)),
            "common_pixels": int(common.sum()),
            "mae_mm": float(np.mean(abs(error))) if len(error) else None,
            "rmse_mm": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
            "p95_abs_mm": float(np.percentile(abs(error), 95)) if len(error) else None}


def save_comparison(folder, stamp, frame, calibration, custom, realsense, source,
                    progress=None):
    started = perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f"{stamp}_{source}"

    def step(message):
        LOGGER.info("comparison save %s stamp=%s folder=%s", message, stamp, folder)
        if progress:
            progress(message)

    step("현재 방식 PLY 저장 중 (1/3)")
    custom_count = save_aligned_point_cloud(f"{prefix}_custom.ply", custom, frame, calibration)
    step("rs.align PLY 저장 중 (2/3)")
    rs_count = save_aligned_point_cloud(f"{prefix}_rs_align.ply", realsense, frame, calibration)
    common = (custom > 0) & (realsense > 0)
    error = abs(custom - realsense) * 1000
    level = np.uint8(np.clip(error * 255 / 10, 0, 255))
    colors = cv2.cvtColor(cv2.applyColorMap(level, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colors[~common] = 0
    step("차이 PLY 저장 중 (3/3)")
    difference_count = save_aligned_point_cloud(
        f"{prefix}_difference.ply", np.where(common, custom, 0), frame, calibration, colors)
    metrics = comparison_metrics(custom, realsense)
    metrics.update({"calibration_source": source, "custom_points": custom_count,
                    "rs_align_points": rs_count, "difference_points": difference_count,
                    "difference_color_scale_mm": [0, 10]})
    scale = calibration_values(frame, calibration)[4]
    step("PNG와 JSON 저장 중")
    _write_image(f"{prefix}_custom_depth.png", np.uint16(np.round(custom / scale)))
    _write_image(f"{prefix}_rs_align_depth.png", np.uint16(np.round(realsense / scale)))
    _write_image(f"{prefix}_difference_0-10mm.png", cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))
    Path(f"{prefix}_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    step(f"완료 ({perf_counter() - started:.1f}초)")
    return metrics
