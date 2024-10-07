import torch
import cv2
import numpy as np
from torchvision.transforms import Compose
from utils.dpt.models import DPTDepthModel
from utils.dpt.transforms import Resize, NormalizeImage, PrepareForNet

def load_model(model_type="dpt_hybrid_nyu", model_path=None, device=None):
    """
    Load the model with the specified type and return it.
    """
    if model_path is None:
        model_path = "utils/weights/dpt_hybrid_nyu-2ce69ec7.pt"
        

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "dpt_hybrid_nyu":  # DPT-Hybrid NYU model
        net_w = 640
        net_h = 480
        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    transform = Compose(
        [
            Resize(
                net_w, net_h, resize_target=None, keep_aspect_ratio=True,
                ensure_multiple_of=32, resize_method="minimal", 
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    return model, transform, device

def depth_map_from_frame(frame, model, transform, device):
    """
    Compute the depth map from a single image frame.
    
    Args:
        frame: The input image frame (numpy array).
        model: The pre-loaded DPT model.
        transform: The transformation pipeline for the model.
        device: The device on which the model runs (CPU or GPU).
    
    Returns:
        depth_map_cv2: The computed depth map in OpenCV-compatible format.
    """
    img_input = transform({"image": frame})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    # Scale for dpt_hybrid_nyu and convert to OpenCV format
    depth_map = prediction * 1000.0  # for dpt_hybrid_nyu, scale depth

    # Normalize the depth map to 0-255 for visualization
    depth_map_cv2 = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_cv2 = depth_map_cv2.astype(np.uint8)

    return depth_map_cv2
