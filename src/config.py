import os

def create_environment(run_folders):
    """
    Crate the base folders for the project

    Input:
        run_folders: dict, containing the base folders for the project
            Example:
                run_folders = {
                    "model_path": "./experiments/checkpoints/",
                    "logs_path": "./experiments/logs/",
                    "sample_path": "./experiments/samples/",
                }

    Returns:
        None
    """
    try:
        os.mkdir(run_folders["model_path"])
    except:
        pass
    try:
        os.mkdir(run_folders["logs_path"])
    except:
        pass
    try:
        os.mkdir(run_folders["sample_path"])
    except:
        pass

    # Preparing required I/O paths for each experiment
    if len(os.listdir(run_folders["model_path"])) == 0:
        exp_idx = 1
    else:
        exp_idx = len(os.listdir(run_folders["model_path"])) + 1

    exp_name = f"exp_{exp_idx:04d}"

    exp_model_folder = os.path.join(run_folders["model_path"], exp_name)
    exp_logs_folder = os.path.join(run_folders["logs_path"], exp_name)
    exp_sample_folder = os.path.join(run_folders["sample_path"], exp_name)

    try:
        os.mkdir(exp_model_folder)
    except:
        pass
    try:
        os.mkdir(exp_logs_folder)
    except:
        pass
    