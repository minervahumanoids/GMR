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


def compute_scaled_human_lowest_z(smplx_frame_data_list, retargeter):
    human_lowest_z = []
    for frame_data in smplx_frame_data_list:
        human_frame = {
            body_name: [np.asarray(pos), np.asarray(quat)]
            for body_name, (pos, quat) in frame_data.items()
        }
        scaled_human = retargeter.scale_human_data(
            human_frame, retargeter.human_root_name, retargeter.human_scale_table
        )
        lowest_z = min(pos[2] for pos, _ in scaled_human.values())
        human_lowest_z.append(lowest_z)
    return np.asarray(human_lowest_z, dtype=np.float32)


def snap_near_ground_to_zero(target_z, threshold_m=0.03):
    snapped = np.asarray(target_z, dtype=np.float32).copy()
    snapped[snapped < threshold_m] = 0.0
    return snapped


def get_grounding_body_indices(kinematics_model):
    contact_indices = [
        index for index, body_name in enumerate(kinematics_model.body_names)
        if body_name.endswith("_contact_point")
    ]
    return contact_indices or None


def apply_height_adjust(root_pos, root_rot, dof_pos, kinematics_model, device, mode: str, human_lowest_z=None):
    body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )
    grounding_body_indices = get_grounding_body_indices(kinematics_model)
    grounding_body_pos_z = (
        body_pos[:, grounding_body_indices, 2]
        if grounding_body_indices is not None
        else body_pos[..., 2]
    )
    ground_offset = 0.0
    if mode == "clip":
        lowest_height = torch.min(grounding_body_pos_z).item()
        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
        return
    if mode == "frame":
        lowest_heights = torch.min(grounding_body_pos_z, dim=1).values.detach().cpu().numpy()
        root_pos[:, 2] = root_pos[:, 2] - lowest_heights + ground_offset
        return
    if mode == "human_frame":
        if human_lowest_z is None:
            raise ValueError("human_lowest_z is required for height_adjust_mode='human_frame'")
        lowest_heights = torch.min(grounding_body_pos_z, dim=1).values.detach().cpu().numpy()
        target_floor_z = snap_near_ground_to_zero(human_lowest_z)
        root_pos[:, 2] = root_pos[:, 2] - lowest_heights + target_floor_z
        return
    raise ValueError(f"Unknown height_adjust_mode: {mode}")


def make_result(status, smplx_file_path, tgt_file_path, error=None):
    return {
        "status": status,
        "smplx_file_path": smplx_file_path,
        "tgt_file_path": tgt_file_path,
        "error": error,
    }


def process_file(smplx_file_path, tgt_file_path, tgt_robot, SMPLX_FOLDER, tgt_folder, fk_device, use_collision_avoidance, height_adjust_mode, total_files, verbose=False):
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
            error = "Memory usage is still high after 10 pauses."
            print(f"[ERROR] {smplx_file_path}: {error}")
            return make_result("failed", smplx_file_path, tgt_file_path, error)

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
        mocap_frame_rate = smplx_data["mocap_frame_rate"]
        log_memory("After loading SMPL-X data")
    except Exception as e:
        error = f"Error loading {smplx_file_path}: {e}"
        print(error)
        return make_result("failed", smplx_file_path, tgt_file_path, str(e))
    
  
    tgt_fps = 30
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as e:
        error = f"Error processing {smplx_file_path}: {e}"
        print(error)
        return make_result("failed", smplx_file_path, tgt_file_path, str(e))
    
    # retarget
    retargeter = GMR(
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        use_collision_avoidance=use_collision_avoidance,
    )
    human_lowest_z = None
    if height_adjust_mode == "human_frame":
        human_lowest_z = compute_scaled_human_lowest_z(smplx_frame_data_list, retargeter)
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
        error = f"Error processing {smplx_file_path}: {e}"
        print(error)
        return make_result("failed", smplx_file_path, tgt_file_path, str(e))
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
    
    HEIGHT_ADJUST = True
    if HEIGHT_ADJUST:
        apply_height_adjust(
            root_pos,
            root_rot,
            dof_pos,
            kinematics_model,
            device,
            height_adjust_mode,
            human_lowest_z=human_lowest_z,
        )
        
    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]
        
        
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
    }


    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
        
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

    return make_result("succeeded", smplx_file_path, tgt_file_path)


def process_file_star(args):
    return process_file(*args)


def write_failure_log(tgt_folder, failed_results):
    if not failed_results:
        return None
    failure_log_path = os.path.join(tgt_folder, "_retarget_failures.txt")
    with open(failure_log_path, "w") as f:
        for result in failed_results:
            f.write(f"{result['smplx_file_path']}\t{result['error']}\n")
    return failure_log_path
    


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
        "--enable-foot-collision-avoidance",
        action="store_true",
        help="Enable collision-aware IK using robot-specific collision_avoidance config, e.g. T1 foot-foot avoidance.",
    )
    parser.add_argument(
        "--height-adjust-mode",
        choices=["clip", "frame", "human_frame"],
        default="clip",
        help="Ground-normalize using a single whole-clip shift, independent per-frame shifts, or the scaled human lowest-point trajectory.",
    )
    args = parser.parse_args()
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    fk_device = resolve_torch_device(args.device)
    print(f"Using FK device: {fk_device}")

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions_paths = [hard_motions_folder / "0.txt", 
                          hard_motions_folder / "1.txt"]
    hard_motions = []
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
                        args.enable_foot_collision_avoidance,
                        args.height_adjust_mode,
                    ))
    print("full args_list:", len(args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    
    new_args_list = []
    for arguments in args_list:
        motion_name = arguments[0].split("/")[-1].split('.')[0]
        if motion_name in hard_motions:
            continue
        if any(content in motion_name for content in exclude_file_content):
            continue
        new_args_list.append(arguments)
    args_list = new_args_list
    
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    work_items = [args + (total_files, verbose) for args in args_list]
    succeeded = 0
    failed = 0
    failed_results = []

    def handle_result(index, result):
        nonlocal succeeded, failed
        if result["status"] == "succeeded":
            succeeded += 1
            print(f"[OK] {index}/{total_files}: {result['tgt_file_path']}")
            return
        failed += 1
        failed_results.append(result)
        print(f"[FAIL] {index}/{total_files}: {result['smplx_file_path']} -> {result['error']}")

    if args.num_cpus <= 1:
        for index, work_item in enumerate(work_items, start=1):
            result = process_file(*work_item)
            handle_result(index, result)
    else:
        pool_context = mp.get_context("spawn") if fk_device.startswith("cuda") else mp
        with pool_context.Pool(args.num_cpus) as pool:
            for index, result in enumerate(pool.imap_unordered(process_file_star, work_items), start=1):
                handle_result(index, result)

    failure_log_path = write_failure_log(tgt_folder, failed_results)
    print(f"Done. Saved to {tgt_folder}")
    print(f"Summary: attempted={total_files}, succeeded={succeeded}, failed={failed}")
    if failure_log_path is not None:
        print(f"Failure log: {failure_log_path}")


if __name__ == "__main__":
    main()
