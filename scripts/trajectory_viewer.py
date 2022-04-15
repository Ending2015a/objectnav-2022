# --- built in ---
import argparse
# --- 3rd party ---
import cv2
import rlchemy
import numpy as np
# --- my module ---


def example(args):
  traj = rlchemy.envs.load_trajectory(args.path)
  cur_frame = 0
  total_frame = len(traj['done'])
  print(traj.keys())

  while True:
    obs = traj['obs']
    cv2.imshow('rgb', np.transpose(obs['rgb'][cur_frame], (1, 2, 0))[...,::-1])
    cv2.imshow('large_map', np.transpose(obs['large_map'][cur_frame], (1, 2, 0))[...,::-1])
    cv2.imshow('small_map', np.transpose(obs['small_map'][cur_frame], (1, 2, 0))[...,::-1])
    print('action: {}, reward: {}, done: {}'.format(
      traj['act'][cur_frame],
      traj['rew'][cur_frame],
      traj['done'][cur_frame]
    ))

    key = cv2.waitKey(0)
    if key == ord('q'):
      exit(0)
    elif key == ord('d'):
      cur_frame = min(cur_frame+1, total_frame-1)
    elif key == ord('a'):
      cur_frame = max(0, cur_frame-1)
    else:
      continue




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", type=str, required=True, help="Trajectory file path")
  args = parser.parse_args()
  example(args)