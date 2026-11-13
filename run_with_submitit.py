# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as detection
import submitit


def parse_args():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for detection", parents=[detection_parser])
    parser.add_argument("--ngpus", default=4, type=int, help="Number of gpus to request on each node (GH200 nodes have 4)")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=60 * 24, type=int, help="Duration of the job in minutes (partition max is 3 days)")
    parser.add_argument("--partition", default="gpu", type=str, help="Slurm partition")
    parser.add_argument("--account", default="naiss2026-3-377-gpu", type=str, help="Slurm account")
    parser.add_argument("--cpus_per_task", default=16, type=int, help="CPU cores per task (one task per GPU)")
    parser.add_argument("--mem_per_gpu", default=100, type=int, help="Memory in GB per GPU")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.set_defaults(
        coco_path="/nobackup/proj/flash/naiss2025-6-383/personal/lingkai/data/coco/zips",
        num_workers=int(os.getenv("DETR_NUM_WORKERS", 8)),
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    # Must be on a filesystem visible from all compute nodes (also used for the dist init file).
    p = Path(os.getenv(
        "DETR_EXP_DIR",
        "/nobackup/proj/flash/naiss2025-6-383/personal/lingkai/experiments/detr",
    ))
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as detection

        self._setup_gpu_args()
        detection.main(self.args)

    def checkpoint(self):
        import os
        import submitit
        from pathlib import Path

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, cluster="slurm", slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    executor.update_parameters(
        mem_gb=args.mem_per_gpu * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        slurm_partition=args.partition,
        slurm_additional_parameters={"account": args.account},
        # each job builds/activates its own aarch64 env, whatever shell submitted it
        slurm_setup=[f"source {Path(__file__).resolve().parent}/setup_env.sh"],
    )

    # submitit would otherwise launch the job with the *submitting* shell's (x86) python.
    # `python` is a SlurmExecutor constructor arg, not an update_parameters one, so set it directly.
    executor._executor.python = os.path.expanduser("~/deep-learning/.venv-aarch64/bin/python")

    executor.update_parameters(name="detr")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
