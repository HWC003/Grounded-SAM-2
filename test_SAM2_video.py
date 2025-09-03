import os
import cv2
import torch
import numpy as np
import supervision as sv
import time

from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
PKG_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VIDEO_DIR = "../test_videos/clipped_scooping_short/"
TEXT_PROMPT = "bowl. food. spoon."
OUTPUT_VIDEO_PATH = "../test_results/sam2/scooping_tracking_demo.mp4"
SAVE_TRACKING_RESULTS_DIR = "../tracking_results/sam2"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]

VIDEO_DIR = os.path.expanduser(VIDEO_DIR)
if not os.path.exists(VIDEO_DIR):
    raise ValueError(f"Video directory {VIDEO_DIR} does not exist")

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("Created SAM image and video predictor from SAM2 model...")
start_t = time.time()

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

print("Created SAM image and video predictor in {:.3f} seconds".format(time.time() - start_t))

# init grounding dino model from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
# print(processor.post_process_grounded_object_detection.__doc__)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(VIDEO_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=VIDEO_DIR)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
"""
# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(VIDEO_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)

# run Grounding DINO on the image
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

# Grounding DINO timing
start_t = time.time()
with torch.no_grad():
    outputs = grounding_model(**inputs)
print(f"Grounding DINO inference on the image in {time.time() - start_t:.3f} seconds")

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.25,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image.convert("RGB")))

# process the detection results
input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W)
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

# # init video predictor state
# # Load first frame
# video_predictor.load_first_frame(image)

PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""

video_segments = {}
frame_times = {}   # dict to store per-frame timings
all_times = []     # or use a list if you only need order

for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    start_t = time.time()

    # store segmentation results
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    elapsed = time.time() - start_t
    frame_times[out_frame_idx] = elapsed
    all_times.append(elapsed)

    print(f"Frame {out_frame_idx}: video predictor took {elapsed:.3f}s")

# After loop → summary stats
total_time = sum(all_times)
avg_time = total_time / len(all_times)
fps = len(all_times) / total_time

print(f"\nProcessed {len(all_times)} frames")
print(f"Total inference time: {total_time:.3f}s")
print(f"Average per frame: {avg_time:.3f}s ({fps:.2f} FPS)")

    

"""
Step 5: Visualize the segment results across the video and save them
"""

save_dir = SAVE_TRACKING_RESULTS_DIR

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(VIDEO_DIR, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


"""
Step 6: Convert the annotated frames to video
"""

create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
