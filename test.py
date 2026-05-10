from dataset.tool.h5_2_lerobotev3 import  H5Dataset

if __name__ == "__main__":
    input_path = "data/train_episode/wipe_board/wipe_board/episode_0002_20260427_110738.h5"
    import h5py
    import numpy as np
    h5reader = H5Dataset(input_path,h5py=h5py,np=np,max_episodes=1)
    h5reader.inspect()