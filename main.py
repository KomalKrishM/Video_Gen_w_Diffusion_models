
import torch
import numpy as np
import imageio
import av
from diffusers import DiffusionPipeline

def load_video_frames(path, target_fps=8, size=(256, 256), duration_sec=40):
    """
    Loads and resizes video frames from the first `duration_sec` seconds.
    """
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_image().resize(size).convert("RGB")
        frames.append(np.array(img))
        if len(frames) >= target_fps * duration_sec:
            break
    return np.array(frames)

def save_video(frames, path="generated_video.mp4", fps=8):
    """
    Saves a NumPy array of frames as an MP4 video.
    """
    writer = imageio.get_writer(path, fps=fps)
    for img in frames:
        writer.append_data(img)
    writer.close()

# def main():
# === Config ===
input_path = "short_video.mp4"  # your 2-second video
prompt_text = "Ambience of the room which i am sitting"
output_path = "extended_output.mp4"
fps = 8
num_generated_frames = 240  # ~30 seconds of generated video
size = (256, 256)

# === Load pretrained pipeline ===
model_id = "cerspense/zeroscope_v2_576w"
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# === Load short video clip ===
print("Loading input video...")
input_frames = load_video_frames(input_path, target_fps=fps, size=size)
print(f"Loaded {len(input_frames)} frames.")

# === Generate video ===
print("Generating extended video with prompt...")
output = pipe(
    prompt=prompt_text,
    reference_video=input_frames,
    num_generated_frames=num_generated_frames
)
generated = output.videos[0]  # shape: (frames, H, W, 3)

# === Save output video ===
print(f"Saving output to {output_path}")
save_video(generated, path=output_path, fps=fps)
print("Saved the generated video.!")

# if __name__ == "__main__":
#     main()

