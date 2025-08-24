import modal
import os
from pathlib import Path

# --- Configuration ---
MODAL_NFS_NAME = "lora-edit-data"
MODAL_NFS_ROOT = Path("/data")
REPO_DIR = Path("/root/LoRAEdit")

# Define the Modal Stub
stub = modal.Stub("lora-edit")

# --- Model Download Function ---
# This function runs *once* during the image build process to download
# the Florence-2 model into the standard Hugging Face cache directory.
def download_florence_model():
    from transformers import AutoProcessor, AutoModelForCausalLM
    model_id = "multimodalart/Florence-2-large-no-flash-attn"
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# --- Container Image Definition ---
# This defines the environment where the code will run.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone --recurse-submodules https://github.com/cjeen/LoRAEdit.git /root/LoRAEdit",
        "cd /root/LoRAEdit && pip install -r requirements.txt",
        "cd /root/LoRAEdit && pip install xformers",
        "pip install huggingface_hub typer",
        # Download the Wan2.1-I2V model
        "huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /root/models/Wan2.1-I2V-14B-480P --local-dir-use-symlinks False",
        # Download the SAM2 model
        "mkdir -p /root/models/models_sam && wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O /root/models/models_sam/sam2_hiera_large.pt",
    )
    .run_function(download_florence_model)
)

# Define a persistent Network File System to store processed data and results
shared_volume = modal.NetworkFileSystem.persisted(MODAL_NFS_NAME)

# --- Train Function ---
# This function runs the LoRA training on a GPU.
@stub.function(
    image=image,
    gpu="A10G",  # Request a specific GPU type
    network_file_systems={str(MODAL_NFS_ROOT): shared_volume},
    timeout=7200,  # Set a 2-hour timeout for the training job
)
def train(sequence_name: str):
    """
    Runs the LoRA training process on a specified data sequence.

    Args:
        sequence_name: The name of the sequence directory inside the NFS,
                       which contains the preprocessed data.
    """
    import subprocess

    print(f"Starting training for sequence: {sequence_name}")

    # Define paths
    data_dir = MODAL_NFS_ROOT / sequence_name
    config_path = data_dir / "configs" / "training.toml"
    lora_output_dir = data_dir / "lora"

    # Verify that the preprocessed data exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found at {config_path}. "
            f"Please ensure the directory '{sequence_name}' exists in the NFS "
            "and contains the output from the preprocessing step."
        )

    print(f"Using training config: {config_path}")

    # Construct and run the training command
    cmd = [
        "deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "--config", str(config_path)
    ]
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"

    process = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("--- Training Output ---")
    print(process.stdout)
    print("-----------------------")

    if process.returncode == 0:
        print("✅ Training process completed successfully.")
        # Check for a final LoRA file as a sign of success
        if any(lora_output_dir.glob("**/adapter_model.safetensors")):
            print("Found trained LoRA adapter file.")
        else:
            print("⚠️ Warning: Training finished, but could not find a LoRA adapter file.")
    else:
        print(f"❌ Training process failed with return code {process.returncode}.")
        raise RuntimeError("Training failed. Check the logs for details.")

# --- Inference Function ---
# This function runs the inference process on a GPU.
@stub.function(
    image=image,
    gpu="A10G",
    network_file_systems={str(MODAL_NFS_ROOT): shared_volume},
    timeout=1800,  # Set a 30-minute timeout
)
def inference(sequence_name: str):
    """
    Runs the inference process to generate the final edited video.

    Args:
        sequence_name: The name of the sequence directory inside the NFS.
                       This directory must contain the trained LoRA and an
                       'edited_image.png' file uploaded by the user.
    """
    import subprocess

    print(f"Starting inference for sequence: {sequence_name}")

    # Define paths
    data_dir = MODAL_NFS_ROOT / sequence_name
    edited_image_path = data_dir / "edited_image.png"
    final_video_path = data_dir / "edited_video.mp4"
    wan_model_path = "/root/models/Wan2.1-I2V-14B-480P" # Path inside the container

    # Verify that the edited image exists
    if not edited_image_path.exists():
        raise FileNotFoundError(
            f"Edited image not found at {edited_image_path}. "
            "Please upload your 'edited_image.png' to the "
            f"'{sequence_name}' directory in the '{MODAL_NFS_NAME}' NFS volume "
            "before running inference."
        )

    print(f"Using data directory: {data_dir}")

    # Construct and run the inference command
    cmd = [
        "python", "inference.py", "--model_root_dir", wan_model_path, "--data_dir", str(data_dir)
    ]

    process = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("--- Inference Output ---")
    print(process.stdout)
    print("------------------------")

    if process.returncode == 0 and final_video_path.exists():
        print(f"✅ Inference process completed successfully.")
        print(f"Final video saved to '{final_video_path}' in the '{MODAL_NFS_NAME}' volume.")
    else:
        print(f"❌ Inference process failed.")
        raise RuntimeError("Inference failed. Check the logs for details.")

# --- Main CLI Block ---
# This allows the user to run the functions from the command line using 'modal run'.
@stub.local_entrypoint()
def main(
    sequence_name: str = "my_awesome_video",
    skip_train: bool = False,
    skip_inference: bool = False,
):
    """
    Main entrypoint to run the LoRA-Edit workflow on Modal.

    How to use:
    1.  First, run the local preprocessing UI to generate your data:
        > python predata_app.py

        In the UI, ensure the 'Data Processing Save Path' is 'processed_data/<sequence_name>'
        and the 'Model Checkpoint Path' is '/root/models/Wan2.1-I2V-14B-480P'.

    2.  Upload your processed data directory to the Modal NFS:
        > modal nfs put lora-edit-data processed_data/my_awesome_video /my_awesome_video
        (Replace 'my_awesome_video' with your actual sequence_name if you changed it)

    3.  Run both training and inference on Modal:
        > modal run modal_app.py --sequence-name my_awesome_video

    4.  To run only inference on an already trained model:
        a. First, upload your edited first frame:
           > modal nfs put lora-edit-data path/to/your/edited_image.png /my_awesome_video/edited_image.png
        b. Then run the script with --skip-train:
           > modal run modal_app.py --sequence-name my_awesome_video --skip-train
    """
    if not skip_train:
        print(f"🚀 Starting remote training job for '{sequence_name}'...")
        train.remote(sequence_name)

    if not skip_inference:
        print(f"🚀 Starting remote inference job for '{sequence_name}'...")
        inference.remote(sequence_name)

    print("✅ All jobs complete.")
