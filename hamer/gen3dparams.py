# this version save 2d, 3d keypoints, handpose and hand vertices
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

def adjust_bbox_coordinates(bbox, image_shape):
    x, y, w, h = bbox
    img_width, img_height = image_shape

    # Adjust x-coordinate if out of bounds
    x = max(0, min(x, img_width - 1))

    # Adjust y-coordinate if out of bounds
    y = max(0, min(y, img_height - 1))

    # Adjust width and height if out of bounds
    w = min(w, img_width - x)
    h = min(h, img_height - y)

    return round(x), round(y), round(w), round(h)

def flatten_dict(data_dict, parent_key='', sep='_'):
    items = []
    for k, v in data_dict.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def remove_index(data, index):
    if isinstance(data, torch.Tensor):
        return torch.cat((data[:index], data[index+1:]), dim=0)
    elif isinstance(data, dict):
        return {k: remove_index(v, index) for k, v in data.items()}
    else:
        raise ValueError("Unsupported data type")

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--data_type', type=str, default=None, choices=["COCO", "13k", "hico", "LAION"], help='Folder with input images')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=True, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=True, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--part_len', type=int, default=-1, help='Folder with input images')
    parser.add_argument('--idx_part', type=int, default=0, help='Folder with input images')

    args = parser.parse_args()
    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    
    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    # print("loading hamer model t/ake: ", time.time()-s)
    # Load detector

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)
    # print("loading human model take: ", time.time()-s)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer      
    if (args.data_type == "COCO"):
        img_folder = "/lustre/scratch/client/vinai/users/ngannh9/hand/data/COCOWholeBody/train2017"
    elif args.data_type == "hico":
        img_folder = "/lustre/scratch/client/vinai/users/ngannh9/hand/data/halpe/hico_20160224_det/images/train2015"
    elif args.data_type == "13k":
        img_folder = "/lustre/scratch/client/vinai/users/ngannh9/hand/data/hand13k/training_dataset/training_data/images"
    elif args.data_type == "LAION":
        img_folder = "/lustre/scratch/client/vinai/users/ngannh9/hand/data/LAION/preprocessed_2256k/train"
    else:
        print("don't recognize data to infering")
    # Make output directory if it does not exist
    out_folder = os.path.join(os.path.join("output", args.data_type), "part" + str(args.idx_part))
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "fully_output"), exist_ok=True)
    idx_part = args.idx_part
    idx_part = 0 if args.part_len == -1 else idx_part
    start_idx = idx_part*args.part_len
    if args.data_type == "LAION":
        with open("/lustre/scratch/client/vinai/users/ngannh9/hand/LAVIS/human.txt", 'r') as file:
            lines = file.readlines()
        img_files = [os.path.join(img_folder, line.strip()) for line in lines]
        # Get all demo images ends with .jpg or .png
        img_paths = sorted(img_files)[start_idx:start_idx+args.part_len]
    else:
        img_paths = sorted([img for end in args.file_type for img in Path(img_folder).glob(end)])[start_idx:start_idx+args.part_len]
    print("infering for ", args.data_type, ", from: ", start_idx, " to: ", start_idx+args.part_len)
    # Iterate over all images in folder
    for img_path in tqdm(img_paths, desc='INFERING...: '):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)

        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)


        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            out = flatten_dict(out)
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            multiplier = (2*batch['right']-1)

            # Render the result
            batch_size = batch['img'].shape[0]
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            out['box_center'] = batch['box_center']
            out['box_size'] = batch['box_size']
            for n in range(batch_size):
                # Get filename from path img_path
                is_right = batch['right'][n].cpu().numpy()
                out['pred_vertices'][n] = (2*is_right-1)*out['pred_vertices'][n]
            out_path = os.path.join(os.path.join(out_folder, "fully_output"), f'{img_fn}.pth')
            torch.save(out, out_path)

    print("Finished. Results written to ", out_folder)
if __name__ == '__main__':
    main()
