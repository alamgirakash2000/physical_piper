# AgileX Piper Robot Modality Configuration for GR00T N1.6
# This file is loaded by launch_finetune.py to register the Piper embodiment.
# Follows the same pattern as oxe_droid in gr00t/configs/data/embodiment_configs.py

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.configs.data.embodiment_configs import register_modality_config

# Piper robot config: 6-DOF arm + gripper
# State: observation.state = [joint1..joint6, gripper] (7D)
# Action: action = [joint_delta1..joint_delta6, gripper_action] (7D)
# Video: global (overhead) + wrist cameras

PIPER_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["global", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_positions",
            "gripper_position",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # action horizon of 16
        modality_keys=[
            "joint_positions",
            "gripper_position",
        ],
        action_configs=[
            # joint_positions (6D) - relative deltas
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_position (1D) - absolute position
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

# Register on import (this is what launch_finetune.py expects)
register_modality_config(PIPER_MODALITY_CONFIG, EmbodimentTag.NEW_EMBODIMENT)
