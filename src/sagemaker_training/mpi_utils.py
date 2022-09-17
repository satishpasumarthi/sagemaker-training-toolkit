import argparse
from inspect import getfile, isclass
import os
import gethostname
import subprocess
import time

import psutil

from sagemaker_training import logging_config,

logger = logging_config.get_logger()


mpi_basic_options = dict()
mpi_basic_options["map-by"] = "slot"
mpi_basic_options["bind-to"] = "none"
mpi_basic_options["allow-run-as-root"] = None
mpi_basic_options["tag-output"] = None

mpi_mca_options = dict()
mpi_mca_options["plm_rsh_no_tree_spawn"] = 1
mpi_mca_options["orte_abort_on_non_zero_status"] = 1
mpi_mca_options["btl_vader_single_copy_mechanism"] = "none"
mpi_mca_options["btl_tcp_if_include"] = self._network_interface_name
mpi_mca_options["oob_tcp_if_include"] = self._network_interface_name
mpi_mca_options["btl"] = "^openlib"
mpi_mca_options["pml"] = "ob1"

mpi_env_options = dict()
mpi_env_options["NCCL_MIN_RINGS"] = 4
mpi_env_options["NCCL_SOCKET_IFNAME"] = self._network_interface_name
mpi_env_options["NCCL_DEBUG"] = "INFO"
mpi_env_options["LD_PRELOAD"] = getfile(gethostname)
mpi_env_options["LD_LIBRARY_PATH"] = None
mpi_env_options["PATH"] = None

mpi_efa_options = dict()
mpi_efa_options["FI_PROVIDER"] = "efa"
mpi_efa_options["FI_EFA_USE_DEVICE_RDMA"] = 1
mpi_efa_options["NCCL_PROTO"] = "simple"


def parse_custom_mpi_options(custom_mpi_options):
    """Parse custom MPI options provided by user. Known options default value will be overridden
    and unknown options will be identified separately."""

    parser = argparse.ArgumentParser()

    for options_dict in [mpi_basic_options, mpi_env_options, mpi_efa_options]:
        for key in options_dict.keys():
            value = options_dict[key]
            parser.add_argument(f"--{key}", default=value, type=type(value))

    known_args, unknown_args = parser.parse_known_args(custom_mpi_options.split())
    return known_args, unknown_args

def write_env_vars_to_file():  # type: () -> None
    with open("/etc/environment", "a") as f:
        for name in os.environ:
            f.write("{}={}\n".format(name, os.environ.get(name)))

def on_terminate(proc):
    logger.info("Invoked on_terminate from psutil.wait_for_procs")
    logger.info("process {} terminated with exit code {}".format(proc, proc.returncode))


def wait_orted_process_to_finish():  # type: () -> None
    orted = orted_process()
    logger.info("Orted process found %s", orted)
    logger.info("Waiting for orted process %s", orted)
    gone, alive = psutil.wait_procs(orted, callback=on_terminate)
    return gone, alive


def orted_process():  # pylint: disable=inconsistent-return-statements
    """Wait a maximum of 5 minutes for orted process to start."""
    for _ in range(5 * 60):
        procs = [p for p in psutil.process_iter(attrs=["name"]) if p.info["name"] == "orted"]
        if procs:
            logger.info("Process[es]: %s", procs)
            return procs
        time.sleep(1)


def write_status_file(host, status_file):
    try:
        logger.info(f"Start writing mpirun finished status to {host}")
        output = subprocess.run(
            ["ssh", str(host), "touch", f"{status_file}"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"output from subprocess run {output}")
        logger.info("Finished writing status file")
        return True
    except subprocess.CalledProcessError:
        logger.info(f"Cannot connect to {host}")
        return False