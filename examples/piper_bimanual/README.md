# Piper Bimanual LongVLA Data Pipeline

This example directory contains the data tooling for collecting raw Piper episodes
for LongVLA and converting them into LeRobot format.

Recommended workflow:

1. Start the Piper ROS stack and camera topics.
2. Run `data_collection/publish_filtered_joint_feedback.py` to republish filtered joint feedback.
3. Run `data_collection/record_longvla_hdf5.py` to save a full raw episode into HDF5.
4. Run `conversion/convert_longvla_hdf5_to_lerobot.py` to create a compact LeRobot dataset
   containing only the fields currently needed by LongVLA.

The HDF5 raw format is intentionally richer than the final LeRobot format so we can
reconvert later with different task definitions or feature choices.
