"""
Integration tests: Running pipeline stages sequentially with temporary test fixtures.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from anime_character_organizer.workflows.audit import run_dataset_audit
from anime_character_organizer.workflows.cluster import run_clustering
from anime_character_organizer.workflows.crop import run_crop_preparation
from anime_character_organizer.workflows.materialize import run_folder_materialization


class TestPipelineWorkflowsIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.input_dir = self.base_dir / "input_images"
        self.input_dir.mkdir()
        self.project_dir = self.base_dir / "pipeline_workspace"
        self.project_dir.mkdir()

        # Create 10 visually distinct synthetic test images
        for i in range(10):
            img = Image.new("RGB", (128, 128), color=(255, 255, 255))
            draw = Image.new("RGB", (32, 32), color=(i * 25, (255 - i * 25), (i * 50) % 255))
            img.paste(draw, (i * 8, (9 - i) * 8))
            img.save(self.input_dir / f"sample_{i:03d}.png")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_audit_crop_cluster_materialize_pipeline(self):
        # 1. Run Stage 1 Audit
        audit_res = run_dataset_audit(
            input_dir=self.input_dir,
            project_dir=self.project_dir,
            show_progress=False,
        )
        self.assertTrue(audit_res["run_dir"].exists())
        self.assertEqual(len(audit_res["valid_df"]), 10)

        # 2. Run Stage 2 Crop Preparation
        crop_res = run_crop_preparation(
            project_dir=self.project_dir,
            previous_run_dir=audit_res["run_dir"],
            use_perceptual_representatives=False,
            show_progress=False,
        )
        self.assertTrue(crop_res["run_dir"].exists())
        self.assertEqual(len(crop_res["crops_df"]), 10)
        self.assertEqual(crop_res["summary"]["successful_crops"], 10)

        # 3. Simulate Stage 3 by providing synthetic embedding run
        sim_embed_run = self.project_dir / "runs" / "03_ccip_embeddings_simulated"
        (sim_embed_run / "arrays").mkdir(parents=True)
        (sim_embed_run / "tables").mkdir(parents=True)

        crops_df = crop_res["crops_df"]
        emb_manifest = crops_df.copy()
        emb_manifest["embedding_row"] = np.arange(len(crops_df), dtype=int)
        emb_manifest.to_csv(sim_embed_run / "tables" / "successful_embedding_manifest.csv", index=False)

        # 10 synthetic 16D embeddings forming two clusters
        np.random.seed(42)
        c1 = np.random.normal(loc=1.0, scale=0.01, size=(5, 16)).astype(np.float32)
        c2 = np.random.normal(loc=-1.0, scale=0.01, size=(5, 16)).astype(np.float32)
        sim_embeddings = np.vstack([c1, c2])
        # normalize
        sim_embeddings /= np.linalg.norm(sim_embeddings, axis=1, keepdims=True)
        np.save(sim_embed_run / "arrays" / "ccip_embeddings_l2.npy", sim_embeddings)

        # 4. Run Stage 4 Clustering
        cluster_res = run_clustering(
            project_dir=self.project_dir,
            previous_run_dir=sim_embed_run,
            min_cluster_size=3,
            min_samples=2,
            show_progress=False,
        )
        self.assertTrue(cluster_res["run_dir"].exists())
        self.assertEqual(len(cluster_res["cluster_manifest_df"]), 10)

        # 5. Run Stage 5 Materialization (using copy mode to be safe across filesystems)
        mat_res = run_folder_materialization(
            project_dir=self.project_dir,
            previous_run_dir=cluster_res["run_dir"],
            materialization_mode="copy",
            show_progress=False,
        )
        self.assertTrue(mat_res["run_dir"].exists())
        self.assertTrue(mat_res["final_output_dir"].exists())
        self.assertEqual(len(mat_res["result_df"]), 10)
        self.assertTrue(mat_res["validation_df"]["is_valid"].all())
        self.assertTrue((mat_res["final_output_dir"] / "README.md").exists())
        self.assertTrue((mat_res["final_output_dir"] / "_global_materialization_index.csv").exists())


if __name__ == "__main__":
    unittest.main()
