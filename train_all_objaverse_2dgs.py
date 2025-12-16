import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Configuration
OBJ_ROOT = "/leonardo_work/IscrC_GEN-X3D/GS/ShapeSplat-Gaussian_MAE/render_scripts/objaverse_renders"
TRAIN_SCRIPT = "train.py"  # 2DGS training script
RENDER_SCRIPT = "render.py"  # 2DGS rendering script

# How many runs in parallel on this node
MAX_WORKERS = 1  # Set >1 ONLY if you have multiple GPUs on the node

def run_one(scene_path: str, gpu_id: int = 0, 
            export_mesh: bool = False, output_base: str = None,
            skip_train: bool = False, iteration: int = -1,
            voxel_size: float = -1.0, depth_trunc: float = -1.0,
            sdf_trunc: float = -1.0, num_cluster: int = 50,
            unbounded: bool = False, mesh_res: int = 1024):
    """
    Run complete pipeline for a single scene: training + optional mesh export.
    """
    scene_name = os.path.basename(scene_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Step 1: Training (unless skipped)
    if not skip_train:
        print(f"\n{'='*60}")
        print(f"[GPU {gpu_id}] Training scene: {scene_name}")
        print(f"{'='*60}")
        scene_name = os.path.basename(scene_path)
        model_dir = f"/leonardo_work/IscrC_GEN-X3D/GS/2d-gaussian-splatting/output/{scene_name}"
        # Build training command
        train_cmd = ["python", TRAIN_SCRIPT, "-s", scene_path, "-m", model_dir]
        
        print(f"Training command: {' '.join(train_cmd)}")
        
        # Run training
        train_result = subprocess.run(train_cmd, env=env)
        
        if train_result.returncode != 0:
            print(f"✗ Training failed for {scene_name} (code: {train_result.returncode})")
            return scene_path, train_result.returncode, "training"
        
        print(f"✓ Training completed for {scene_name}")
    
    print(f"Model path: {model_dir}")
    
    # Step 3: Mesh export (if requested)
    if export_mesh:
        print(f"\n[GPU {gpu_id}] Exporting mesh for: {scene_name}")
        
        # Build render command for mesh export
        render_cmd = [
            "python", RENDER_SCRIPT,
            "-m", model_dir,
            "-s", scene_path,
            "--skip_train",  # Skip training image rendering
            "--skip_test",   # Skip test image rendering
        ]
        
        # Add iteration if specified
        if iteration > 0:
            render_cmd.extend(["--iteration", str(iteration)])
        
        # Add mesh export parameters
        render_cmd.extend(["--voxel_size", str(voxel_size)])
        render_cmd.extend(["--depth_trunc", str(depth_trunc)])
        render_cmd.extend(["--sdf_trunc", str(sdf_trunc)])
        render_cmd.extend(["--num_cluster", str(num_cluster)])
        
        if unbounded:
            render_cmd.append("--unbounded")
            render_cmd.extend(["--mesh_res", str(mesh_res)])
        
        print(f"Mesh export command: {' '.join(render_cmd)}")
        
        # Run mesh export
        render_result = subprocess.run(render_cmd, env=env)
        
        if render_result.returncode != 0:
            print(f"✗ Mesh export failed for {scene_name} (code: {render_result.returncode})")
            return scene_path, render_result.returncode, "mesh_export"
        
        print(f"✓ Mesh export completed for {scene_name}")
        
        # Check if mesh files were created
        train_dir = os.path.join(model_dir, 'train', f"ours_{iteration if iteration > 0 else 30000}")
        if os.path.exists(train_dir):
            ply_files = [f for f in os.listdir(train_dir) if f.endswith('.ply')]
            print(f"Created {len(ply_files)} mesh file(s): {ply_files}")
    
    return scene_path, 0, "success"

def read_ids(path: str):
    """Read UIDs from file."""
    with open(path, "r") as f:
        for line in f:
            uid = line.strip()
            if uid:
                yield uid

def main(uid_file: str, export_mesh: bool = False, output_base: str = None,
         skip_train: bool = False, limit: int = None, offset: int = 0,
         **mesh_params):
    """
    Main function to process multiple scenes.
    
    Args:
        uid_file: Path to file containing UIDs
        export_mesh: Whether to export mesh after training
        output_base: Base directory for model outputs
        skip_train: Skip training phase (only do mesh export)
        limit: Limit number of scenes to process
        offset: Skip first N scenes
        mesh_params: Parameters for mesh export
    """
    # Read all UIDs
    all_uids = list(read_ids(uid_file))
    print(f"Loaded {len(all_uids)} UIDs from {uid_file}")
    
    all_uids = all_uids[324:]
  
    
    print(f"Processing {len(all_uids)} scenes (offset={offset}, limit={limit})")
    
    # Build scene paths
    scene_paths = []
    for uid in all_uids:
        scene_dir = os.path.join(OBJ_ROOT, uid)
        if os.path.isdir(scene_dir):
            # Check if scene has required structure (e.g., images folder)
            images_dir = os.path.join(scene_dir, "image")
            if os.path.isdir(images_dir):
                scene_paths.append(scene_dir)
            else:
                print(f"Warning: No 'images' directory in {scene_dir}")
        else:
            print(f"Warning: {scene_dir} not found")
    
    print(f"Found {len(scene_paths)} valid scenes to process")
    
    if not scene_paths:
        print("No valid scenes found. Exiting.")
        return
    
    # Create output base directory if specified
    if output_base and not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
    
    # Determine available GPUs
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
    print(f"Using {NUM_GPUS} GPU(s)")
    
    # Run processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        
        for i, scene_path in enumerate(scene_paths):
            gpu_id = i % NUM_GPUS
            
            # Submit job with all parameters
            future = executor.submit(
                run_one, 
                scene_path, 
                gpu_id, 
                export_mesh, 
                output_base,
                skip_train,
                mesh_params.get('iteration', -1),
                mesh_params.get('voxel_size', -1.0),
                mesh_params.get('depth_trunc', -1.0),
                mesh_params.get('sdf_trunc', -1.0),
                mesh_params.get('num_cluster', 50),
                mesh_params.get('unbounded', False),
                mesh_params.get('mesh_res', 1024)
            )
            futures[future] = scene_path
        
        # Process results
        results = []
        for future in as_completed(futures):
            scene_path = futures[future]
            scene_name = os.path.basename(scene_path)
            
            try:
                _, return_code, stage = future.result()
                if return_code == 0:
                    result_msg = f"✓ Success: {scene_name} ({stage})"
                    print(result_msg)
                    results.append((scene_name, "success", stage))
                else:
                    result_msg = f"✗ Failed (code {return_code}): {scene_name} ({stage})"
                    print(result_msg)
                    results.append((scene_name, "failed", stage))
            except Exception as e:
                result_msg = f"✗ Exception for {scene_name}: {str(e)}"
                print(result_msg)
                results.append((scene_name, "exception", str(e)))
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    success_count = sum(1 for _, status, _ in results if status == "success")
    print(f"Successfully processed: {success_count}/{len(results)}")
    
    if success_count < len(results):
        print("\nFailed scenes:")
        for scene_name, status, stage in results:
            if status != "success":
                print(f"  - {scene_name}: {status} ({stage})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch train 2DGS on multiple scenes with optional mesh export"
    )
    
    # Core arguments
    parser.add_argument("--uid_file", type=str, required=True,
                       help="Path to file containing UIDs (one per line)")
    
    # Training control
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip training phase (use existing model)")
    parser.add_argument("--output_base", type=str, default=None,
                       help="Base directory for model outputs")
    
    # Mesh export control
    parser.add_argument("--export_mesh", action="store_true",
                       help="Export mesh after training", default=True)
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Model iteration to load (default: latest)")
    
    # Mesh parameters (from render.py)
    parser.add_argument("--voxel_size", type=float, default=-1.0,
                       help="Voxel size for TSDF (default: auto)")
    parser.add_argument("--depth_trunc", type=float, default=-1.0,
                       help="Max depth range for TSDF (default: auto)")
    parser.add_argument("--sdf_trunc", type=float, default=-1.0,
                       help="Truncation value for TSDF (default: auto)")
    parser.add_argument("--num_cluster", type=int, default=50,
                       help="Number of connected clusters to export")
    parser.add_argument("--unbounded", action="store_true",
                       help="Use unbounded mode for meshing")
    parser.add_argument("--mesh_res", type=int, default=1024,
                       help="Resolution for unbounded mesh extraction")
    
    # Processing control
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of scenes to process")
    parser.add_argument("--offset", type=int, default=0,
                       help="Skip first N scenes")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Update MAX_WORKERS
    MAX_WORKERS = args.workers
    
    # Change to 2DGS directory
    os.chdir("/leonardo_work/IscrC_GEN-X3D/GS/2d-gaussian-splatting")
    
    # Prepare mesh parameters
    mesh_params = {
        'iteration': args.iteration,
        'voxel_size': args.voxel_size,
        'depth_trunc': args.depth_trunc,
        'sdf_trunc': args.sdf_trunc,
        'num_cluster': args.num_cluster,
        'unbounded': args.unbounded,
        'mesh_res': args.mesh_res
    }
    
    # Run main function
    main(
        uid_file=args.uid_file,
        export_mesh=args.export_mesh,
        output_base=args.output_base,
        skip_train=args.skip_train,
        limit=args.limit,
        offset=args.offset,
        **mesh_params
    )