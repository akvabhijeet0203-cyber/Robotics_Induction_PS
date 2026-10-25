"""Smoke tests for the explainable PS2 computer-vision baseline."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool_sorting import (cartesian_to_polar, create_demo_images, detect_tools, estimate_homography,
                          match_template_variants, pixel_to_table, process_image,
                          table_to_robot, _classify_shape, _watershed_masks)


class ToolSortingTests(unittest.TestCase):
    def test_demo_contains_each_expected_class(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "outputs") as temp:
            image_path = create_demo_images(Path(temp))[0]
            import cv2
            image = cv2.imread(str(image_path))
            detections, masks = detect_tools(image)
            self.assertEqual(set(masks), {"red_proposal", "blue_proposal", "green_proposal"})
            self.assertEqual({d.label for d in detections}, {"wrench", "caliper", "cable"})
            self.assertTrue(all(0 <= d.center_px[0] < image.shape[1] for d in detections))

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_tools(np.zeros((12, 12), np.uint8))

    def test_planar_homography_maps_pixels(self):
        h = np.array([[2, 0, 10], [0, 3, -5], [0, 0, 1]], dtype=float)
        self.assertEqual(pixel_to_table((4, 2), h), (18.0, 1.0))

    def test_homography_estimation_and_robot_transform(self):
        image_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
        table_points = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], np.float32)
        h = estimate_homography(image_points, table_points)
        self.assertTrue(np.allclose(pixel_to_table((50, 50), h), (1, 0.5)))
        transform = np.eye(4)
        transform[0, 3] = 0.4
        self.assertTrue(np.allclose(table_to_robot((1, 0.5), transform), (1.4, 0.5, 0.0)))
        with self.assertRaises(ValueError):
            estimate_homography(image_points[:3], table_points[:3])
        radius, bearing = cartesian_to_polar((3, 4))
        self.assertAlmostEqual(radius, 5.0)
        self.assertAlmostEqual(bearing, np.arctan2(4, 3))

    def test_grabcut_mode_finds_separated_demo_shapes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "outputs") as temp:
            image_path = create_demo_images(Path(temp))[0]
            import cv2
            image = cv2.imread(str(image_path))
            detections, masks = detect_tools(image, segmentation="grabcut")
            self.assertIn("foreground", masks)
            self.assertEqual({d.label for d in detections}, {"wrench", "caliper", "cable"})

    def test_watershed_can_propose_a_split(self):
        import cv2
        foreground = np.zeros((140, 180), np.uint8)
        cv2.circle(foreground, (72, 70), 39, 255, -1)
        cv2.circle(foreground, (108, 70), 39, 255, -1)
        image = np.full((140, 180, 3), 220, np.uint8)
        masks = _watershed_masks(image, foreground)
        self.assertGreaterEqual(len(masks), 2)

    def test_unknown_shape_is_not_forced_into_a_tool_class(self):
        import cv2
        shape = np.zeros((180, 180), np.uint8)
        cv2.rectangle(shape, (30, 60), (150, 120), 255, -1)
        contour = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        self.assertEqual(_classify_shape(contour)[0], "unknown")

    def test_process_writes_visualization_and_coordinates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "outputs") as temp:
            root = Path(temp)
            image_path = create_demo_images(root / "input")[0]
            detections = process_image(image_path, root / "out")
            self.assertTrue(detections)
            self.assertTrue((root / "out" / "demo_01_separated_tools_analysis.png").is_file())
            self.assertTrue((root / "out" / "demo_01_separated_tools_detections.json").is_file())

    def test_template_matching_finds_exact_unrotated_patch(self):
        rng = np.random.default_rng(5)
        template = rng.integers(0, 255, (24, 20), dtype=np.uint8)
        image = np.full((90, 100), 255, np.uint8)
        image[31:55, 47:67] = template
        matches = match_template_variants(image, template, scales=(1.0,), angles=(0,), threshold=0.85)
        self.assertTrue(matches)
        self.assertEqual(matches[0]["bbox_xywh"], (47, 31, 20, 24))


if __name__ == "__main__":
    unittest.main()
