import logging
import os
import sys
import json
from timeit import default_timer as timer
import resource

logging.basicConfig(level=logging.INFO)


def save_script(output_path, args=None):
    """Save the script and arguments for reproducibility"""

    logging.info(f"Saving script and arguments at {output_path}")

    this_script = sys.argv[0]
    with open(this_script, "r") as cur_file:
        cur_running = cur_file.readlines()
    with open(os.path.join(output_path, this_script), "w") as log_file:
        log_file.writelines(cur_running)

    # get script name without extension
    script_name = os.path.splitext(os.path.basename(this_script))[0]

    with open(os.path.join(output_path, script_name + "_args.txt"), "w") as log_file:
        log_file.writelines([arg + "\n" for arg in sys.argv])

    if args is not None:
        with open(os.path.join(output_path, script_name + "_args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)


def get_output_path(
    output_dir,
    output_prefix,
    args,
    defaults,
    always_include=None,
):
    if output_prefix is None:
        name = ""
    else:
        name = output_prefix
    included = set()

    arg_format = "_{}={}"

    if always_include is not None:
        for arg in always_include:
            name += arg_format.format(arg, args[arg])
            included.add(arg)

    for arg in defaults:
        if arg in included:
            continue
        if args[arg] != defaults[arg]:
            name += arg_format.format(arg, args[arg])

    output_path = os.path.join(
        output_dir,
        name,
    )

    return output_path


class Timer:
    def __init__(self):
        self.last_time = timer()

    def usage_and_time(self, message=""):
        message = f"{message} Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB, Time: {timer() - last_time:.2f} s"
        self.last_time = timer()
        return message


timer = Timer()
