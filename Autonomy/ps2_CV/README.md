# PS2: Computer Vision for Robotic Object Sorting

This project is a classical, explainable starting point for the IVDC robotic tool-sorting task. It demonstrates preprocessing, edge display, foreground segmentation, shape-template matching, localization, candidate grasp points, and calibrated coordinate mapping. It does not command a robot.

## Method

1. Retain the color image. Apply a small Gaussian blur; derive grayscale and Canny edge views for inspection.
2. Form foreground proposals with either HSV color ranges (the reproducible illustrations) or GrabCut (a mostly uniform background).
3. Optionally apply marker-based watershed to propose splits for touching foreground regions. Inspect the result: watershed can over-segment or merge tools.
4. Clean binary masks with morphological opening/closing and extract external contours.
5. Compare each candidate silhouette with three centered, scale-normalized tool templates across rotations. Keep `unknown` if the best silhouette overlap is too weak. This is a small hand-built classifier, not a trained model.
6. Use contour moments and PCA for pixel center/orientation; mark a candidate grasp point. A centroid is only a visualization and may be unsafe.
7. Optionally map pixels to table coordinates with a calibrated homography. Robot-base coordinates require an additional calibrated table-to-robot transform.

The attached examples are generated illustrations, not photographs from the Roboflow dataset. They are useful for inspecting the pipeline but do not establish accuracy on real tools.

## Install

From `Autonomy/ps2_CV`:

```powershell
python -m pip install -r requirements.txt
```

## Run

Generate and analyze both included illustrations:

```powershell
python tool_sorting.py --make-demo
```

Analyze an image against a mostly uniform background:

```powershell
python tool_sorting.py --input path/to/tools.jpg --segmentation grabcut
```

Try foreground splitting for touching objects:

```powershell
python tool_sorting.py --input path/to/tools.jpg --segmentation watershed
```

Build the report and regenerate demo outputs:

```powershell
python build_report.py
```

Each run writes a labeled stage visualization and JSON/CSV detections under `outputs/`.

## Calibration

`estimate_homography(image_points, table_points)` accepts at least four corresponding, non-collinear points in pixel and table units. `pixel_to_table` applies the resulting 3 x 3 matrix. For a robot-base target, also apply `table_to_robot` with a measured 4 x 4 transform. `cartesian_to_polar` reports distance and bearing relative to a chosen planar origin. The repository has no fabricated calibration values: measure the actual table and robot setup before using physical coordinates.

## Files

- `tool_sorting.py` - preprocessing, HSV/GrabCut/Watershed proposals, contour features, silhouette matching, template matching, calibration helpers, CLI, and figures.
- `research.md` - numbered explanations, method, observations, limitations, and sources.
- `PS2_tool_sorting_report.pdf` - designed report with workflow and example visualizations.
- `demo_images/` - reproducible separated and overlapping illustrations.
- `outputs/` - plots and machine-readable detections generated from those illustrations.
- `test_tool_sorting.py` - focused pipeline and coordinate tests.

## Limitations

The included silhouette templates cover only three simplified shapes. Real products vary in shape, color, wear, lighting, viewpoint, and occlusion. GrabCut assumes a useful foreground/background contrast; watershed requires good markers. Similarity distance is not a probability. Grasp selection needs geometry, collision, reachability, and gripper checks. The demo outputs should be replaced or supplemented with licensed real dataset images before claiming dataset performance.

## Acknowledgement

AI assistance was used to draft and review code and documentation. Official OpenCV references and the assignment-linked dataset listing are cited in `research.md`. Review each threshold and limitation so you can explain the approach yourself.
