## 数据处理
- h5数据初步滤波
```
python data_process/tool/filter_h5_butterworth.py --input-dir ../nero_ws/runs/bg_data --output-dir ../nero_ws/runs/bg_data_filted --cutoff-hz 15 --dataset teleop/q_follower --datas
et teleop/dq_follower --dataset teleop/tau_follower --dataset teleop/q_cmd
```
- h5 -> lerobotv3 隔帧降采样
```
python data_process/tool/downsample_h5_2_lerobotv3.py --config config/shape_meta/data/next_data_dual_phase.yaml
```
