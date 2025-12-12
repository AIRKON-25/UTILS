#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import depthai as dai


def parse_args():
    parser = argparse.ArgumentParser(description="Display DepthAI camera output.")
    parser.add_argument("--fps", type=float, default=10.0, help="Camera FPS (default: 10)")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Directory to store captured frames. Disabled when not provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(sensorFps=args.fps)
        video_queue = cam.requestOutput(
            (1536, 864),
            resizeMode=dai.ImgResizeMode.LETTERBOX,
            enableUndistortion=True,
        ).createOutputQueue()

        pipeline.start()
        frame_idx = 0
        while pipeline.isRunning():
            video_in = video_queue.get()
            assert isinstance(video_in, dai.ImgFrame)
            frame = video_in.getCvFrame()
            cv2.imshow("video", frame)

            if save_dir is not None:
                frame_path = save_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_idx += 1

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
