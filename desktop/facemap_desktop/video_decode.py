"""Bundled, local-file video decoding for the desktop and workspace workers."""

import math
from pathlib import Path
import time

import av
import cv2
import numpy as np
from PIL import Image


def _open(source):
    # A file object cannot be interpreted as a URL by FFmpeg.
    return av.open(str(Path(source).resolve()), options={'protocol_whitelist': 'file,pipe'}, timeout=30)


def _stream(container):
    streams = container.streams.video
    if len(streams) != 1 or streams[0].disposition & av.stream.Disposition.attached_pic:
        raise ValueError('Use an ordinary single-camera video with one video track')
    stream = streams[0]
    stream.codec_context.thread_count = 2
    return stream


def _info(container, stream, frame=None):
    codec = stream.codec_context
    duration = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration or 0) / av.time_base
    ratio = stream.sample_aspect_ratio
    return {'streams': [{'codec_type': 'video', 'codec_name': codec.name,
        'width': codec.width, 'height': codec.height,
        'duration': duration, 'avg_frame_rate': str(stream.average_rate or 0),
        'pix_fmt': codec.format.name if codec.format else '',
        'sample_aspect_ratio': f'{ratio.numerator}:{ratio.denominator}' if ratio else '1:1',
        'color_transfer': {16: 'smpte2084', 18: 'arib-std-b67'}.get(codec.color_trc, str(codec.color_trc)),
        'color_space': str(codec.colorspace),
        'side_data_list': [{'rotation': frame.rotation if frame is not None else 0}]}]}


def _checked_frames(container, stream, metadata=None):
    from .video_input import MAX_FRAMES, MAX_DURATION
    first = previous = None
    started = time.monotonic()
    for index, frame in enumerate(container.decode(stream)):
        if index >= MAX_FRAMES or time.monotonic() - started > 240:
            raise ValueError('Video decoding exceeded its frame or time limit')
        timestamp = frame.time
        if timestamp is None or not math.isfinite(timestamp) or (previous is not None and timestamp <= previous):
            raise ValueError('Video needs finite, strictly increasing frame timestamps')
        if first is None:
            first = timestamp
        if timestamp - first > MAX_DURATION:
            raise ValueError('Use a capture no longer than 30 seconds')
        previous = timestamp
        if metadata and (frame.width != metadata['width'] or frame.height != metadata['height']
                         or frame.rotation != metadata['rotation_degrees']):
            raise ValueError('Video dimensions or rotation changed while decoding')
        if frame.format.name not in ('yuv420p', 'yuvj420p', 'yuv422p', 'yuvj422p', 'yuv444p', 'yuvj444p', 'rgb24', 'bgr24'):
            raise ValueError('This workflow requires 8-bit SDR video')
        yield index, timestamp - first, frame


def probe(source):
    from .video_input import validate_stream
    with _open(source) as container:
        stream = _stream(container)
        metadata = validate_stream(_info(container, stream))
        times = []
        first_time = None
        for index, timestamp, frame in _checked_frames(container, stream):
            if index == 0:
                metadata = validate_stream(_info(container, stream, frame))
                first_time = frame.time
            if (frame.width, frame.height, frame.rotation) != (metadata['width'], metadata['height'], metadata['rotation_degrees']):
                raise ValueError('Video dimensions or rotation changed while decoding')
            times.append(timestamp)
        if not times:
            raise ValueError('Video contains no decodable frames')
        metadata.update(first_timestamp_seconds=first_time, decoded_frame_count=len(times))
        return metadata, times


def _upright(frame, *, proxy=False):
    # PyAV exposes FFmpeg's counterclockwise display rotation but does not apply it.
    rotation = frame.rotation
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > .001:
        raise ValueError('Only standard right-angle display rotations are supported')
    if proxy:
        scale = min(1., 960 / max(frame.width, frame.height))
        frame = frame.reformat(width=max(2, round(frame.width * scale)),
                               height=max(2, round(frame.height * scale)), format='gray', interpolation='AREA')
        pixels = frame.to_ndarray()
    else:
        pixels = frame.to_ndarray(format='rgb24')
    return np.ascontiguousarray(np.rot90(pixels, round(rotation / 90)))


def analyze(source, metadata, times, region, progress):
    from .video_input import analyze_frame
    detector = cv2.ORB_create(nfeatures=700)
    candidates = []
    with _open(source) as container:
        for index, timestamp, frame in _checked_frames(container, _stream(container), metadata):
            if index >= len(times) or abs(timestamp - times[index]) > .0001:
                raise ValueError('Decoded frame timestamps changed during processing')
            candidates.append(analyze_frame(_upright(frame, proxy=True), index, timestamp, region, detector))
            if index % 30 == 0:
                progress({'phase': 'analyzing', 'frame': index, 'total_frames': len(times)})
    if len(candidates) != len(times):
        raise ValueError('Decoded frame count changed during processing')
    return candidates


def export(source, metadata, times, selected, images):
    wanted = {candidate.index: index for index, candidate in enumerate(selected)}
    count = 0
    with _open(source) as container:
        for index, timestamp, frame in _checked_frames(container, _stream(container), metadata):
            if index >= len(times) or abs(timestamp - times[index]) > .0001:
                raise ValueError('Decoded frame timestamps changed during export')
            if index in wanted:
                Image.fromarray(_upright(frame)).save(images / f'frame_{wanted[index]:04d}.png', compress_level=3)
                count += 1
    if count != len(selected):
        raise ValueError('Exported image count differs from the selection')
