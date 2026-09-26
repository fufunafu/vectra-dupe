"""Local, uncalibrated video-to-photo benchmark input. Never creates a scan session.

Analysis thumbnails are disposable in-memory proxies. Exported PNGs retain the
decoded video's pixel dimensions (with display rotation), without resizing.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw

VERSION = 1
MAX_DURATION = 31.0  # Allow container rounding around a 30-second recording.
MAX_FRAMES = 2000
MAX_PIXELS = 16_777_216
MAX_SOURCE_BYTES = 2 * 1024**3
MAX_EXPORT_BYTES = 8 * 1024**3
MIN_IMAGES = 12
DEFAULT_REGION = (0.15, 0.10, 0.70, 0.80)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def meets_4k(width, height):
    return max(width, height) >= 3840 and min(width, height) >= 2160


def validate_region(region):
    if (len(region) != 4 or not all(math.isfinite(v) for v in region)
            or min(region[:2]) < 0 or min(region[2:]) <= 0
            or region[0] + region[2] > 1 or region[1] + region[3] > 1):
        raise ValueError("Subject region must be normalized x,y,width,height inside the image")
    if min(region[2:]) < 0.1:
        raise ValueError("Subject region is too small for reliable analysis")


def _probe(args):
    command = ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", *args, "-of", "json"]
    result = subprocess.run(command, capture_output=True, check=True, timeout=120)
    return json.loads(result.stdout)


def validate_stream(info):
    videos = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if len(videos) != 1 or videos[0].get("disposition", {}).get("attached_pic"):
        raise ValueError("Use an ordinary single-camera video with one video track")
    stream = videos[0]
    width, height = int(stream["width"]), int(stream["height"])
    if not meets_4k(width, height):
        raise ValueError(f"Video is {width}x{height}; native 4K or higher is required. No upscaling is performed")
    if width > 8192 or height > 8192 or width * height > MAX_PIXELS:
        raise ValueError("Video exceeds this local test path's pixel budget")
    duration = float(stream.get("duration", info.get("format", {}).get("duration", "nan")))
    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Video has no usable frame-rate metadata") from exc
    if not math.isfinite(duration) or not 0 < duration <= MAX_DURATION:
        raise ValueError("Use a capture no longer than 30 seconds (31-second container tolerance)")
    if not math.isfinite(fps) or not 1 <= fps <= 60.1:
        raise ValueError("Unsupported frame rate; record at 30 FPS")
    if stream.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A"):
        raise ValueError("Anamorphic video is not supported; use square pixels")
    side_data = stream.get("side_data_list", [])
    if (stream.get("color_transfer") in ("smpte2084", "arib-std-b67")
            or any("DOVI" in d.get("side_data_type", "").upper() for d in side_data)
            or stream.get("pix_fmt") not in ("yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "rgb24", "bgr24")):
        raise ValueError("This first test path requires 8-bit SDR video. Record with HDR Video off; HDR/10-bit needs a separately verified color conversion")
    rotation = float(next((d["rotation"] for d in side_data if "rotation" in d),
                          stream.get("tags", {}).get("rotate", 0)))
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 0.001:
        raise ValueError("Only standard right-angle display rotations are supported")
    if round(rotation / 90) % 2:
        display_width, display_height = height, width
    else:
        display_width, display_height = width, height
    return {"width": width, "height": height, "display_width": display_width,
            "display_height": display_height, "rotation_degrees": rotation,
            "duration_seconds": duration, "average_fps": fps,
            "codec": stream.get("codec_name"), "pixel_format": stream["pix_fmt"],
            "color_transfer": stream.get("color_transfer"), "color_space": stream.get("color_space")}


def probe_video(path):
    metadata = validate_stream(_probe(["-show_streams", "-show_format", "-i", str(path)]))
    frames = _probe(["-select_streams", "v:0", "-show_frames", "-show_entries",
                     "frame=best_effort_timestamp_time,width,height", "-i", str(path)]).get("frames", [])
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("Video frame count exceeds the bounded test input or contains no frames")
    times = [float(f.get("best_effort_timestamp_time", "nan")) for f in frames]
    if (not all(math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))
            or times[-1] - times[0] > MAX_DURATION):
        raise ValueError("Video needs finite, strictly increasing frame timestamps within 30 seconds")
    if any((f.get("width"), f.get("height")) != (metadata["width"], metadata["height"]) for f in frames):
        raise ValueError("Resolution changes within the video are not supported")
    metadata["first_timestamp_seconds"] = times[0]
    metadata["decoded_frame_count"] = len(frames)
    return metadata, [t - times[0] for t in times]


@contextmanager
def _decoder(command, timeout=180):
    # A watchdog bounds pipe reads too, not only the eventual process wait.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        timer = threading.Timer(timeout, process.kill)
        timer.daemon = True
        timer.start()
        try:
            yield process
            if process.wait(timeout=10) != 0:
                errors.seek(0)
                raise RuntimeError("Video decoding failed: " + errors.read(3000).decode(errors="replace"))
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def _ffmpeg_input(source):
    return ["ffmpeg", "-nostdin", "-v", "error", "-n", "-protocol_whitelist", "file,pipe",
            "-threads", "2", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn"]


@dataclass
class Candidate:
    index: int
    time: float
    metrics: dict
    points: np.ndarray
    descriptors: np.ndarray | None
    thumbnail: np.ndarray
    crop_size: tuple

    @property
    def usable(self):
        return not self.metrics["warnings"]


def analyze_frame(gray, index, time, region, detector):
    height, width = gray.shape
    x, y, w, h = region
    crop = gray[round(y * height):round((y + h) * height), round(x * width):round((x + w) * width)]
    if min(crop.shape) < 32:
        raise ValueError("Subject region is too small in the analysis image")
    pixels = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA).astype(float)
    lap = 4 * pixels[1:-1, 1:-1] - pixels[:-2, 1:-1] - pixels[2:, 1:-1] - pixels[1:-1, :-2] - pixels[1:-1, 2:]
    sharpness = float(lap.var())
    mean = float(pixels.mean())
    clipped = float(((pixels <= 5) | (pixels >= 250)).mean())
    warnings = []
    if sharpness < 8:
        warnings.append("blur_or_low_texture")
    if not 20 <= mean <= 235 or clipped >= 0.5:
        warnings.append("exposure")
    keypoints, descriptors = detector.detectAndCompute(crop, None)
    points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    thumbnail = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    return Candidate(index, time, {"sharpness": round(sharpness, 3), "mean_luma": round(mean, 3),
                     "clipped_fraction": round(clipped, 5), "warnings": warnings},
                     points, descriptors, thumbnail, (crop.shape[1], crop.shape[0]))


def analyze_video(source, metadata, times, region, progress):
    scale = min(1, 960 / max(metadata["display_width"], metadata["display_height"]))
    width = max(2, round(metadata["display_width"] * scale))
    height = max(2, round(metadata["display_height"] * scale))
    command = _ffmpeg_input(source) + ["-vf", f"scale={width}:{height}:flags=area,format=gray",
        "-fps_mode", "passthrough", "-frames:v", str(len(times) + 1), "-f", "rawvideo", "pipe:1"]
    detector = cv2.ORB_create(nfeatures=700)
    candidates = []
    with _decoder(command) as process:
        for index, timestamp in enumerate(times):
            remaining, chunks = width * height, []
            while remaining:
                chunk = process.stdout.read(remaining)
                if not chunk:
                    raise ValueError("Decoder produced fewer frames than the timestamp index")
                chunks.append(chunk)
                remaining -= len(chunk)
            gray = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(height, width)
            candidates.append(analyze_frame(gray, index, timestamp, region, detector))
            if index % 100 == 0:
                progress({"phase": "analyzing", "frames": index + 1, "total": len(times)})
        if process.stdout.read(1):
            raise ValueError("Decoder produced more frames than the timestamp index")
    return candidates


def overlap(a, b):
    """Conservative 2D feature continuity proxy, not calibrated 3D overlap."""
    result = {"status": "unverified", "matches": 0, "inliers": 0}
    if a.descriptors is None or b.descriptors is None or min(len(a.points), len(b.points)) < 12:
        return result
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a.descriptors, b.descriptors, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    result["matches"] = len(matches)
    if len(matches) < 12:
        return result
    first = np.float32([a.points[m.queryIdx] for m in matches])
    second = np.float32([b.points[m.trainIdx] for m in matches])
    cv2.setRNGSeed(0)
    transform, mask = cv2.findHomography(first, second, cv2.RANSAC, 3.0)
    if transform is None or mask is None or not np.isfinite(transform).all():
        return result
    mask = mask.ravel().astype(bool)
    result["inliers"] = int(mask.sum())
    if mask.sum() < 12 or mask.mean() < 0.5:
        return result
    w, h = a.crop_size
    coverage = min(cv2.contourArea(cv2.convexHull(points[mask])) / (w * h) for points in (first, second))
    if coverage < 0.08:
        return result
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    projected = cv2.perspectiveTransform(corners[None], transform)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return result
    area = cv2.contourArea(projected)
    if not 0.4 * w * h <= area <= 2.5 * w * h:
        return result
    intersection, _ = cv2.intersectConvexConvex(projected, corners)
    fraction = max(0.0, min(1.0, intersection / max(area, w * h)))
    result.update(status="supported" if fraction >= 0.6 else "weak",
                  estimated_region_overlap=round(fraction, 3), feature_coverage=round(coverage, 3))
    return result


def select_frames(candidates, maximum):
    if not candidates:
        raise ValueError("No decoded video frames")
    # One representative per time window prevents the sharp frontal section
    # from monopolizing selection at the expense of profiles and other views.
    interval = candidates[-1].time + (candidates[-1].time - candidates[-2].time if len(candidates) > 1 else 1)
    buckets = [[] for _ in range(maximum)]
    rows = [{"source_frame_index": c.index, "time_seconds": c.time, **c.metrics,
             "selected": False, "reason": "quality_screen" if not c.usable else "same_time_window"} for c in candidates]
    for c in candidates:
        if c.usable:
            buckets[min(maximum - 1, int(c.time / interval * maximum))].append(c)
    selected, seen = [], set()
    for bucket in buckets:
        ranked = sorted(bucket, key=lambda c: (-c.metrics["sharpness"], c.index))
        options = []
        for c in ranked:
            key = hashlib.sha256(c.thumbnail.tobytes()).digest()
            duplicate = key in seen or (selected and np.abs(c.thumbnail.astype(float) - selected[-1].thumbnail).mean() < 1.0)
            if duplicate:
                rows[c.index]["reason"] = "near_duplicate"
            else:
                options.append(c)
        if not options:
            continue
        chosen = options[0]
        continuity = None
        if selected:
            continuity = overlap(selected[-1], chosen)
            # Prefer demonstrable overlap among comparably sharp candidates.
            for alternative in options[1:4]:
                if continuity["status"] == "supported" or alternative.metrics["sharpness"] < chosen.metrics["sharpness"] * 0.8:
                    break
                check = overlap(selected[-1], alternative)
                if check["status"] == "supported":
                    chosen, continuity = alternative, check
            rows[chosen.index]["previous_selected_overlap"] = continuity
        rows[chosen.index].update(selected=True, reason="sharp_time_representative")
        selected.append(chosen)
        seen.add(hashlib.sha256(chosen.thumbnail.tobytes()).digest())
    return selected, rows


def _contact_sheet(images, selected, output, region):
    indices = sorted(set(np.linspace(0, len(selected) - 1, min(40, len(selected)), dtype=int)))
    sheet = Image.new("RGB", (5 * 240, math.ceil(len(indices) / 5) * 210), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for tile, index in enumerate(indices):
        with Image.open(images / f"frame_{index:04d}.png") as source:
            preview = source.convert("RGB")
            preview.thumbnail((232, 175))
        x, y = tile % 5 * 240 + 4, tile // 5 * 210 + 4
        sheet.paste(preview, (x, y))
        rx, ry, rw, rh = region
        draw.rectangle((x + rx * preview.width, y + ry * preview.height,
                        x + (rx + rw) * preview.width, y + (ry + rh) * preview.height), outline="orange", width=2)
        draw.text((x, y + 178), f"{index:04d} | {selected[index].time:.3f}s", fill="black")
    sheet.save(output, quality=90)


def prepare_video(source, destination, *, maximum=220, region=DEFAULT_REGION, progress=lambda event: None, backend='ffmpeg'):
    validate_region(region)
    if not MIN_IMAGES <= maximum <= 300:
        raise ValueError("Maximum selected images must be between 12 and 300")
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in (".mov", ".mp4", ".m4v"):
        raise ValueError("Provide a local MOV, MP4, or M4V video")
    if not 0 < source.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError("Source video exceeds the 2 GiB test-input limit or is empty")
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output directory already exists; use a new directory to preserve previous work")
    if source.is_relative_to(destination.resolve()):
        raise ValueError("Output must not contain the original video")
    if backend not in ('ffmpeg', 'av'):
        raise ValueError('Unknown video decoder')
    if backend == 'ffmpeg':
        for name in ("ffmpeg", "ffprobe"):
            if shutil.which(name) is None:
                raise ValueError(f"{name} is required")
        decoder_version = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True,
                                         timeout=10, text=True).stdout.splitlines()[0]
    else:
        from . import video_decode
        decoder_version = 'PyAV ' + video_decode.av.__version__
    original_hash = sha256(source)
    metadata, times = probe_video(source) if backend == 'ffmpeg' else video_decode.probe(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # PNG worst-case RGB budget plus overhead, before decoding or writing images.
    budget = min(maximum, len(times)) * metadata["width"] * metadata["height"] * 3 * 1.02
    if budget > MAX_EXPORT_BYTES or shutil.disk_usage(destination.parent).free < budget + 512 * 1024**2:
        raise ValueError("Insufficient disk budget for lossless original-resolution PNGs; reduce --max-frames")
    destination.mkdir(exist_ok=False)
    report = {"format": "facemap-video-input/1", "version": VERSION, "status": "analyzing",
              "tools": {"opencv": cv2.__version__, "numpy": np.__version__,
                        "decoder": decoder_version},
              "display_only": True, "metric_scale_available": False, "camera_calibration_available": False,
              "source": {"path": str(source), "sha256": original_hash, **metadata},
              "settings": {"maximum_selected": maximum, "subject_region": list(region),
                           "region_source": "fixed_region_not_face_detection", "analysis_long_side": 960,
                           "minimum_sharpness": 8, "output_format": "PNG", "export_resizing": False},
              "warnings": ["The region must contain the subject throughout the video; background features can mislead selection.",
                           "2D feature overlap is a heuristic, not proof of matching facial geometry, complete coverage, or accuracy.",
                           "Pixel dimensions do not prove that the source video was not previously upscaled."]}
    manifest = destination / "video_input.json"

    def save_report():
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save_report()
    try:
        cv2.setNumThreads(1)
        candidates = (analyze_video(source, metadata, times, region, progress) if backend == 'ffmpeg'
                      else video_decode.analyze(source, metadata, times, region, progress))
        selected, rows = select_frames(candidates, maximum)
        report["frames"] = rows
        report["summary"] = {"decoded_frames": len(candidates), "usable_candidates": sum(c.usable for c in candidates),
                             "selected_frames": len(selected), "maximum_selected": maximum,
                             "minimum_images": MIN_IMAGES,
                             "unverified_or_weak_pairs": sum(r.get("previous_selected_overlap", {}).get("status", "supported") != "supported" for r in rows)}
        if not 29 <= metadata["average_fps"] <= 31:
            report["warnings"].append("Frame rate differs from the 30 FPS target; original timestamps were retained.")
        gaps = [b.time - a.time for a, b in zip(selected, selected[1:])]
        report["summary"]["largest_selected_gap_seconds"] = max(gaps, default=0)
        if not selected or selected[0].time > 0.5 or times[-1] - selected[-1].time > 0.5 or max(gaps, default=0) > 0.5:
            report["warnings"].append("Selection has temporal coverage gaps over 0.5 seconds; review missing viewpoints.")
        if len(selected) < MIN_IMAGES:
            report["warnings"].append("Too few usable, distinct images for this benchmark. No reconstruction inputs were exported.")
            report["status"] = "insufficient_usable_frames"
        else:
            images = destination / "images"
            images.mkdir()
            expression = "+".join(f"eq(n\\,{c.index})" for c in selected)
            command = _ffmpeg_input(source) + ["-vf", "select=" + expression, "-fps_mode", "passthrough",
                       "-frames:v", str(len(selected)), "-pix_fmt", "rgb24", "-compression_level", "3",
                       "-start_number", "0", str(images / "frame_%04d.png")]
            progress({"phase": "exporting", "frames": len(selected)})
            if backend == 'ffmpeg':
                subprocess.run(command, capture_output=True, check=True, timeout=240)
            else:
                video_decode.export(source, metadata, times, selected, images)
            exported = sorted(images.glob("*.png"))
            if len(exported) != len(selected):
                raise ValueError("Exported image count differs from the selection")
            for index, (file, candidate) in enumerate(zip(exported, selected)):
                with Image.open(file) as image:
                    if image.size != (metadata["display_width"], metadata["display_height"]):
                        raise ValueError("Export dimensions differ from the original video's display dimensions")
                    image.verify()
                rows[candidate.index].update(file=f"images/{file.name}", sha256=sha256(file),
                                             width=metadata["display_width"], height=metadata["display_height"])
            _contact_sheet(images, selected, destination / "contact-sheet.jpg", region)
            report["status"] = "ready_for_review"
        if sha256(source) != original_hash:
            raise ValueError("Original video changed during processing; outputs must not be used")
        report["source_unchanged"] = True
        save_report()
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
        save_report()
        raise
