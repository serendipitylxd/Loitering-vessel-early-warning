import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re, json, argparse
from pathlib import Path
import yaml
from tqdm import tqdm

from mmengine import Config
from mmdet.apis import init_detector, inference_detector

from utils.io_utils import ensure_dir, save_json, list_files_multi

def load_cfg(yaml_path):
    """Load a YAML config file as a Python dict."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def frame_id_from_path(p, rgx):
    """Extract frame id from a filepath using a regex; fallback to filename stem."""
    m = re.match(rgx, Path(p).name)
    if m:
        return m.group(1)
    return Path(p).stem

def run_model(model_cfg, images, labels_cfg, out_dir, frame_rgx):
    """
    Run a single MMDetection model on a list of images and dump detections.

    Assumptions:
      - For MMDet 3.x, inference_detector returns a DetDataSample.
      - model.dataset_meta['classes'] aligns with your dataset label order.
        If not, provide a mapping in your config and use it here.
    """
    name = model_cfg['name']
    cfg_file = model_cfg['config']
    ckpt = model_cfg['checkpoint']
    thr = float(model_cfg.get('score_thr', 0.4))

    print(f"[2D] Loading {name}")
    model = init_detector(cfg_file, ckpt, device='cuda:0')

    lock_gate = labels_cfg['lock_gate']
    results = {}

    for img in tqdm(images, desc=f"[2D:{name}]"):
        out = inference_detector(model, img)
        # MMDet 3.x returns DetDataSample
        inst = out.pred_instances
        bboxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()

        dets = []
        has_gate = False
        for b, s, l in zip(bboxes, scores, labels):
            # Ensure class index -> name mapping matches your dataset; add a mapping if required.
            label_name = model.dataset_meta['classes'][int(l)] if 'dataset_meta' in dir(model) else str(int(l))
            if s < thr:
                continue
            dets.append({
                'bbox_xyxy': [float(x) for x in b.tolist()],
                'score': float(s),
                'label': label_name
            })
            if label_name == lock_gate:
                has_gate = True

        fid = frame_id_from_path(img, frame_rgx)
        results[fid] = {
            'image_path': img,
            'detections': dets,
            'lock_gate_present': has_gate
        }

    save_json(results, os.path.join(out_dir, f'{name}.det2d.json'))
    print(f"[2D] Saved to {os.path.join(out_dir, f'{name}.det2d.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    out_dir = os.path.join(cfg['output']['root'], 'det2d')
    ensure_dir(out_dir)

    # Gather images; supports multiple extensions.
    images = list_files_multi(cfg['data']['images_dir'], ['jpg', 'jpeg', 'png'])
    if not images:
        raise RuntimeError("No images found.")

    frame_rgx = cfg['data']['frame_id_regex']
    labels_cfg = cfg['labels']

    # Iterate over all configured MMDetection models.
    for m in cfg.get('mmdet_models', []):
        run_model(m, images, labels_cfg, out_dir, frame_rgx)

if __name__ == '__main__':
    main()

