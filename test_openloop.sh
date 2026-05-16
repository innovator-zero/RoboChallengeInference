 python3 test_openloop.py \
      --checkpoint openpi/checkpoints/pi05_RC_aloha/pi05_RC_stack_bowls_rel_joint_ms/49999 \
      --robot aloha \
      --task stack_bowls \
      --data_dir /KAIROS_vepfs-2/KAIROS_vepfs/luyuxiang/dataset/RoboChallenge/task_table30_stack_bowls/stack_bowls/data/episode_000000 \
      --exec_horizon 50

#  python3 test_openloop.py \
#       --checkpoint /KAIROS_vepfs-2/KAIROS_vepfs/luyuxiang/code/faster_private/checkpoints/pi05_RC_arx5/pi05_RC_fold_dishcloth_rel_joint/49999 \
#       --robot arx5 \
#       --task fold_dishcloth \
#       --data_dir /KAIROS_vepfs-2/KAIROS_vepfs/luyuxiang/dataset/RoboChallenge/task_table30_fold_dishcloth/fold_dishcloth/data/episode_000000 \
#       --exec_horizon 50