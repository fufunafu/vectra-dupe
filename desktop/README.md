# faceMap Desktop

Import a ZIP from the faceMap iPhone app, reconstruct on your own computer,
inspect the model, and export GLB or PLY. No account, Apple reconstruction SDK,
NVIDIA GPU, processing subscription, or cloud upload is required.

This is a new desktop companion. It does not change the submitted iPhone build
or the existing Mac processing server. Photo-only CPU reconstruction is
experimental and can leave gaps or reject difficult captures. A completed job
means files were produced, not that facial accuracy was validated.

## Start

1. Install **64-bit Python 3.11 or 3.12** from
   [python.org](https://www.python.org/downloads/). Include Tcl/Tk and, on Windows,
   the Python launcher. Python 3.13 and newer are not supported by this package.
2. Extract the desktop package into a normal folder.
3. Windows: double-click `Start-faceMap.cmd`. Mac: open `Start-faceMap.command`.
   Linux: run `sh Start-faceMap.command` in a terminal.
4. First launch installs the engine into a private user environment. Allow
   several minutes, an internet connection, and about 3 GB of free disk space.
   Later processing and viewing work offline.
5. On the iPhone, open **Sessions > Export scan**. Transfer the ZIP to your
   computer using AirDrop to a Mac or a file service accessible from Windows
   or Linux. Choose that ZIP in the desktop window.
6. Choose a result folder and press **Create 3D model**. Then press **View model**.

Each job creates its own new result folder. Existing results and the original
ZIP are not overwritten. Cancel stops the worker and removes its temporary
capture copy. Keep the desktop window open while using the browser viewer.

Linux may require `python3-tk`, `python3-venv`, `libgl1`, and `libgomp1` from the
distribution's package manager. Homebrew Python on Mac requires the matching
`python-tk@3.12` or `python-tk@3.11` package for the window. Command-line
reconstruction works without Tcl/Tk.

## Computers and limits

The intended targets are Windows 10/11 x64, Intel and Apple Silicon Macs, and
mainstream x64 Linux with glibc 2.31 or newer, such as Ubuntu 22.04. The pinned
[Open3D wheels](https://pypi.org/project/open3d/0.19.0/#files) and
[OpenCV wheels](https://pypi.org/project/opencv-python-headless/4.11.0.86/#files)
cover these architectures. Wheel availability alone is not application testing.
See the validation record below for what has actually run.

Start with **Quick** on older machines. An 8 GB RAM computer is the initial
engineering target, not a measured minimum across all captures. The engine
uses at most four native CPU threads, processes photo pairs sequentially,
bounds image size and the number of stereo pairs, and caps output triangles.
Large inputs can still require substantial memory and disk space.

Native Windows ARM, 32-bit systems, ChromeOS without a Linux environment, and
old operating systems are not verified targets. Browser viewing needs WebGL;
the exported model remains usable in another 3D application if WebGL is absent.
The interface reports rendering failures and still exposes the downloads.

## What the modes do

| Mode | Input | Method |
| --- | --- | --- |
| Automatic | Any supported iPhone export | LiDAR depth when present; otherwise photo stereo |
| Depth | At least two usable depth views | CPU TSDF fusion of calibrated depth, with captured-photo texture |
| Photos | At least three overlapping calibrated photos | CPU OpenCV stereo using the iPhone's recorded camera poses, followed by TSDF fusion |

Photo matching checks both directions and rejects unsupported disparity.
It supports horizontal and vertical stereo baselines, including portrait-held
iPhones. It does not reconstruct from arbitrary uncalibrated image folders.
Fast subject movement, blur, lighting changes and camera drift can cause gaps,
distortion, or failed reconstruction. It does not synthesize missing anatomy.

Depth samples are filtered conservatively and the surface receives eight
Taubin smoothing iterations. The report records this processing. No accuracy,
volume, or clinical measurement claims are made. The new CPU photo engine is
not claimed to match Apple Object Capture quality.

## Saved files

- `model.glb`: photo-textured model, in **metres**, centered for viewing.
- `surface-mm.ply`: reconstructed surface in the original **millimetre** frame.
- `report.json`: status, method, frame counts, warnings, processing time, stereo
  pair decisions, smoothing settings, and the native-to-GLB transform.

These files contain a person's face and should be handled like the original
scan. The app never uploads them. The local viewer listens only on `127.0.0.1`,
uses a random session URL, and exposes only the model, report, and bundled
viewer assets. There are no CDN requests or analytics. First-time dependency
installation contacts the Python package registry, without sending scan data.

## Command line

From this directory:

```sh
python3.12 -m venv .venv
# Windows uses .venv\Scripts\python.exe instead of .venv/bin/python.
.venv/bin/python -m pip install --only-binary=:all: .
.venv/bin/python -m facemap_desktop --doctor
.venv/bin/python -m facemap_desktop /path/to/scan.zip --inspect
.venv/bin/python -m facemap_desktop /path/to/scan.zip --output /path/to/new-result --preset quick
.venv/bin/python -m facemap_desktop --view /path/to/new-result
.venv/bin/python -m facemap_desktop --self-test --output /path/to/new-synthetic-result
```

`--mode photos` explicitly tests reconstruction without using the scan's depth.
The synthetic self-test needs no personal photos. `--doctor` distinguishes CPU
engine availability from optional desktop-window availability.

## Verification and native packages

Run from the repository root after installing the package:

```sh
python -m unittest discover -s desktop/tests -v
```

Tests cover nested/root iPhone ZIPs, path traversal, duplicate/case-colliding
names, symbolic links, damaged calibration/depth/images, calibrated stereo
orientation and scale, textureless rejection, full depth and photo-only exports,
output overwrite protection, failed-job cleanup, and local-viewer access limits.

`.github/workflows/desktop.yml` runs this suite on Windows, Linux, Apple Silicon
Mac, and Intel Mac. It builds a PyInstaller folder package on each OS and runs
the synthetic reconstruction with the **packaged executable** before retaining
the artifact. A workflow definition is not evidence that its jobs have passed.
Native packages do not need a separate Python installation. They are not yet
code-signed, notarized, or published as a public download.

Local validation and any remaining platform gates are recorded in
`../docs/desktop-portability-validation-2026-09-26.md`.

Three.js r160 and its loaders are bundled under their MIT license in
`facemap_desktop/viewer/vendor/LICENSE`. Reconstruction dependencies retain
their upstream licenses.
