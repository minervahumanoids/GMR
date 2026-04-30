import argparse
import json
import pathlib
import os
import multiprocessing as mp

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import IK_CONFIG_ROOT
import gc
import time
import psutil
import tracemalloc


def check_memory(threshold_gb=30):  # adjust based on your available memory
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent


def resolve_torch_device(requested_device: str) -> str:
    if requested_device == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda:0")
                return "cuda:0"
            except Exception as exc:
                print(f"[WARNING] CUDA probe failed, falling back to CPU: {exc}")
        return "cpu"
    if requested_device.startswith("cuda"):
        try:
            torch.zeros(1, device=requested_device)
        except Exception as exc:
            raise RuntimeError(f"Requested CUDA device '{requested_device}' is not usable: {exc}") from exc
    return requested_device


def process_file(
    smplx_file_path,
    tgt_file_path,
    tgt_robot,
    SMPLX_FOLDER,
    tgt_folder,
    fk_device,
    robot_xml_path,
    ik_config_path,
    target_fps,
    height_adjust_mode,
    root_origin_mode,
    total_files,
    verbose=False,
):
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
        
    # Initial checks (with optional logging)
    log_memory("Initial memory usage")
    
    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60*2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return False

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
        mocap_frame_rate = smplx_data["mocap_frame_rate"]
        log_memory("After loading SMPL-X data")
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return False
    
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=target_fps,
        )
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return False
    
    # retarget
    retargeter = GMR(
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        robot_xml_path=robot_xml_path,
        ik_config_path=ik_config_path,
    )
    qpos_list = []
    for smplx_frame_data in smplx_frame_data_list:
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())

    qpos_list = np.array(qpos_list)

    log_memory("After retargeting")
    
    device = fk_device
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

    try:
        root_pos = qpos_list[:, :3]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return False
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    log_memory("After forward kinematics")

    body_names = kinematics_model.body_names
    
    if height_adjust_mode == "min_body_to_ground":
        # Optional explicit recipe step: move the lowest body point to z=0.
        body_pos, _ = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
        ground_offset = 0.0
        lowerst_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground
    elif height_adjust_mode != "none":
        raise ValueError(f"Unsupported height_adjust_mode: {height_adjust_mode}")
        
    if root_origin_mode == "zero_xy":
        # Optional explicit recipe step: normalize XY origin to the first frame.
        root_pos[:, :2] -= root_pos[0, :2]
    elif root_origin_mode != "none":
        raise ValueError(f"Unsupported root_origin_mode: {root_origin_mode}")
        
        
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
        "retarget_config": {
            "target_fps": target_fps,
            "height_adjust_mode": height_adjust_mode,
            "root_origin_mode": root_origin_mode,
        },
    }


    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
        
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")
    
    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()
        
    # clean cache
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return True
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
                        )
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=4, type=int)
    parser.add_argument("--device", default="auto", help="Forward-kinematics device: auto, cpu, cuda:0, ...")
    parser.add_argument(
        "--smplx-model-dir",
        default=None,
        help="SMPL-X body model directory containing the smplx/ subfolder. Defaults to GMR assets/body_models.",
    )
    parser.add_argument("--robot-xml-path", default=None, help="Optional robot XML path overriding GMR's robot registry.")
    parser.add_argument("--ik-config-path", default=None, help="Optional IK config JSON path overriding GMR's robot registry.")
    parser.add_argument("--target-fps", default=30, type=int, help="Output FPS used when sampling SMPL-X frames.")
    parser.add_argument(
        "--height-adjust-mode",
        choices=("none", "min_body_to_ground"),
        default="none",
        help="Optional root-z adjustment. Default does no height adjustment.",
    )
    parser.add_argument(
        "--root-origin-mode",
        choices=("none", "zero_xy"),
        default="none",
        help="Optional root XY origin normalization. Default preserves the original root trajectory.",
    )
    parser.add_argument(
        "--filter-hard-motions",
        action="store_true",
        help="Opt in to filtering motions listed in assets/hard_motions/*.txt. Disabled by default.",
    )
    parser.add_argument(
        "--exclude-motion-name-contains",
        action="append",
        default=[],
        metavar="TEXT",
        help="Opt in to skipping motions whose filename stem contains TEXT. May be repeated.",
    )
    args = parser.parse_args()
    if args.target_fps <= 0:
        raise ValueError(f"--target-fps must be positive, got {args.target_fps}")
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    fk_device = resolve_torch_device(args.device)
    print(f"Using FK device: {fk_device}")

    SMPLX_FOLDER = pathlib.Path(args.smplx_model_dir).expanduser().resolve() if args.smplx_model_dir else HERE / ".." / "assets" / "body_models"
    if not (SMPLX_FOLDER / "smplx").is_dir():
        raise FileNotFoundError(f"SMPL-X body model directory not found: {SMPLX_FOLDER / 'smplx'}")
    verbose = False

    hard_motions = []
    if args.filter_hard_motions:
        hard_motions_folder = HERE / ".." / "assets" / "hard_motions"
        hard_motions_paths = [hard_motions_folder / "0.txt",
                              hard_motions_folder / "1.txt"]
        for hard_motions_path in hard_motions_paths:
            with open(hard_motions_path, "r") as f:
                for line in f:
                    if "Motion:" in line:
                        motion_path = line.split(":")[1].strip()
                    else:
                        continue
                    motion_path = motion_path.split(",")[0].strip().split(".")[0]
                    hard_motions.append(motion_path)
                
                
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")
                if not os.path.exists(tgt_file_path) or args.override:
                    args_list.append((
                        smplx_file_path,
                        tgt_file_path,
                        args.robot,
                        SMPLX_FOLDER,
                        tgt_folder,
                        fk_device,
                        args.robot_xml_path,
                        args.ik_config_path,
                        args.target_fps,
                        args.height_adjust_mode,
                        args.root_origin_mode,
                    ))
    print("full args_list:", len(args_list))
    
    # Optional caller-controlled filtering only. By default GMR retargets every input file.
    if hard_motions or args.exclude_motion_name_contains:
        new_args_list = []
        for arguments in args_list:
            motion_name = arguments[0].split("/")[-1].split('.')[0]
            if motion_name in hard_motions:
                continue
            if any(content in motion_name for content in args.exclude_motion_name_contains):
                continue
            new_args_list.append(arguments)
        args_list = new_args_list
    
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    with mp.Pool(args.num_cpus) as pool:
        results = pool.starmap(process_file, [args + (total_files, verbose) for args in args_list])

    # MDP and other callers depend on the process exit status to distinguish
    # "all files retargeted" from "the batch script ran but skipped failures".
    # Without this, one bad SMPL-X file can produce no .pkl while GMR exits 0.
    failed = sum(1 for result in results if not result)
    if failed:
        raise SystemExit(f"GMR failed to process {failed}/{total_files} files.")

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
