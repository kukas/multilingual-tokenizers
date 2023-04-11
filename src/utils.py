import logging
import os
import sys

logging.basicConfig(level=logging.INFO)


def save_script(output_path):
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
