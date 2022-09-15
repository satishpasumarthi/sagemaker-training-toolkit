from __future__ import absolute_import

import psutil

from sagemaker_training import environment, logging_config, params

logger = logging_config.get_logger()

def get_mount_point