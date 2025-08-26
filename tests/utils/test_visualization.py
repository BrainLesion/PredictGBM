import tempfile
import unittest
import numpy as np
from pathlib import Path
from predict_gbm.utils.visualization import (
    get_cmap_norm_patches_tumorseg,
    get_segmentation_projection,
    get_slices,
    grid_plot,
)


class TestVisualizationUtils(unittest.TestCase):
    def test_get_slices_within_bounds(self):
        center = (5, 5, 5)
        axial, coronal = get_slices(
            center, num_slices=3, step_size=1, patient_dim=(10, 10, 10)
        )
        self.assertEqual(axial, [3, 4, 5])
        self.assertEqual(coronal, [3, 4, 5])

        center_edge = (0, 0, 0)
        axial_edge, coronal_edge = get_slices(
            center_edge, num_slices=3, step_size=3, patient_dim=(10, 10, 10)
        )
        self.assertEqual(axial_edge, [0, 0, 0])
        self.assertEqual(coronal_edge, [0, 0, 0])

    def test_get_cmap_norm_patches_tumorseg(self):
        cmap, norm, patches = get_cmap_norm_patches_tumorseg([1, 2, 3])
        self.assertEqual(cmap.N, 4)
        self.assertEqual(len(patches), 3)

    def test_get_segmentation_projection(self):
        seg = np.zeros((2, 2, 2), dtype=int)
        seg[0, 0, 0] = 1
        seg[1, 0, 0] = 1
        projection = get_segmentation_projection(seg, label=1, axis=0)
        expected = np.array([[1, 0], [0, 0]])
        np.testing.assert_array_equal(projection, expected)

    def test_grid_plot_output_exists(self):
        image_tensor = np.empty((1, 2, 2), dtype=object)
        image_tensor[0, 0, 0] = np.zeros((2, 2))
        image_tensor[0, 0, 1] = np.zeros((2, 2))
        image_tensor[0, 1, 0] = np.zeros((2, 2))
        image_tensor[0, 1, 1] = np.zeros((2, 2))
        imshow_args = [{"cmap": "gray"}]
        header = "Header"
        col_titles = ["Col1", "Col2"]
        row_titles = ["Row1", "Row2"]
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "out.pdf"
            grid_plot(
                image_tensor=image_tensor,
                imshow_args=imshow_args,
                header=header,
                col_titles=col_titles,
                row_titles=row_titles,
                outfile=outfile,
            )
            self.assertTrue(outfile.exists())

    def test_grid_plot_dimension_mismatch(self):
        image_tensor = np.empty((1, 2, 2), dtype=object)
        image_tensor[0, 0, 0] = np.zeros((2, 2))
        image_tensor[0, 0, 1] = np.zeros((2, 2))
        image_tensor[0, 1, 0] = np.zeros((2, 2))
        image_tensor[0, 1, 1] = np.zeros((2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "out.pdf"
            with self.assertRaises(ValueError):
                grid_plot(
                    image_tensor=image_tensor,
                    imshow_args=[],  # wrong length
                    header="h",
                    col_titles=["c1", "c2"],
                    row_titles=["r1", "r2"],
                    outfile=outfile,
                )


if __name__ == "__main__":
    unittest.main()
