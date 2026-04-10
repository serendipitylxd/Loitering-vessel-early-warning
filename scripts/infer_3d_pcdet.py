# -*- coding: utf-8 -*-
"""
infer_3d_pcdet.py — Inference script based on OpenPCDet

Functions:
- Read no-stop-lines config.yaml
- Support multiple models (pcdet_models list), run inference sequentially and save outputs as JSON
- Export detections per frame (filtered by score_thr)
"""

import argparse
import json
from pathlib import Path
import glob

import numpy as np
import torch
from pcdet.config import cfg as PC_CFG, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import yaml
from tqdm import tqdm

class DemoDataset(DatasetTemplate):
    """Simple dataset: supports .bin and .npy point cloud files."""
    def __init__(self, dataset_cfg, class_names, root_path, ext=".bin", logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=root_path, logger=logger)
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_dir():
            self.sample_file_list = sorted(glob.glob(str(self.root_path / f"*{self.ext}")))
        else:
            self.sample_file_list = [str(self.root_path)]

        if not self.sample_file_list:
            raise RuntimeError(f"No point files found at {self.root_path} with ext {self.ext}")

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        """Load a single frame of point cloud data."""
        fp = self.sample_file_list[index]
        ext = Path(fp).suffix.lower()

        if ext == ".bin":
            points = np.fromfile(fp, dtype=np.float32).reshape(-1, 4)
        elif ext == ".npy":
            points = np.load(fp)
            # Handle case with only xyz coordinates
            if points.ndim == 2 and points.shape[1] == 3:
                inten = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points.astype(np.float32), inten])
        else:
            raise NotImplementedError(f"Unsupported ext {ext}")

        input_dict = {
            "points": points,
            "frame_id": Path(fp).stem,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def build_label_name_map(class_names):
    """PCDet predicted labels start from 1, map id → class name."""
    return {i + 1: str(n) for i, n in enumerate(class_names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="path to no-stop-lines config.yaml")
    args = parser.parse_args()

    # Load no-stop-lines YAML config
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    points_dir = Path(cfg["data"]["points_dir"])
    out_root = Path(cfg["output"]["root"]) / "det3d"
    out_root.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger()
    logger.info("=========== PCDet Inference from config.yaml ===========")

    # Loop through all PCDet models listed in config
    for model_cfg in cfg.get("pcdet_models", []):
        name = model_cfg["name"]
        cfg_file = model_cfg["config"]
        ckpt = model_cfg["checkpoint"]
        score_thr = model_cfg.get("score_thr", 0.0)

        logger.info(f"[{name}] Loading config={cfg_file}, ckpt={ckpt}")

        # Load PCDet model config
        cfg_from_yaml_file(cfg_file, PC_CFG)

        # Build dataset
        demo_dataset = DemoDataset(
            dataset_cfg=PC_CFG.DATA_CONFIG,
            class_names=PC_CFG.CLASS_NAMES,
            root_path=points_dir,
            ext=".bin",
            logger=logger
        )

        # Build and load model
        model = build_network(model_cfg=PC_CFG.MODEL,
                              num_class=len(PC_CFG.CLASS_NAMES),
                              dataset=demo_dataset)
        model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        id2name = build_label_name_map(PC_CFG.CLASS_NAMES)
        results = {}

        # Run inference on dataset
        with torch.no_grad():
            for idx, data_dict in enumerate(tqdm(demo_dataset, desc=f"[3D:{name}]")):
                frame_id = str(data_dict.get("frame_id", idx))
                batch = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(batch)

                pred_dicts, _ = model.forward(batch)
                pred = pred_dicts[0]

                boxes = pred.get("pred_boxes", torch.empty((0, 7))).cpu().numpy()
                scores = pred.get("pred_scores", torch.empty((0,))).cpu().numpy()
                labels = pred.get("pred_labels", torch.empty((0,), dtype=torch.long)).cpu().numpy().astype(int)

                detections = []
                for b, s, l in zip(boxes, scores, labels):
                    if s < score_thr:
                        continue
                    detections.append({
                        "box7d": [float(t) for t in b.tolist()],
                        "score": float(s),
                        "label_id": int(l),
                        "label": id2name.get(int(l), str(l))
                    })

                results[frame_id] = {
                    "points_path": str(demo_dataset.sample_file_list[idx]),
                    "detections": detections
                }

        # Save results
        out_path = out_root / f"{name}.det3d.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"[{name}] Saved results to {out_path}")

    logger.info("All done.")


if __name__ == "__main__":
    main()

