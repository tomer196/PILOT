import os

data_path = '/home/tomerweiss/Datasets/pd_only/'
sub_lr = 1e-3
rec_lr = 1e-3
acc_weight = 1e-1
vel_weight = 1e-1
batch_size = 30
init = 'spiral_high'  # spiral, EPI, radial, gaussian
n_shots = 16
interp_gap = 1
trajectory_learning = 1
num_epochs = 30
sample_rate = 1.
TSP = ''
SNR = ''

if trajectory_learning == 1:
    test_name = f'{n_shots}/{init}_{sub_lr}_{acc_weight}_{vel_weight}'
else:
    test_name = f'{n_shots}/{init}_fixed'

if TSP == '--TSP':
    test_name = f'{n_shots}/{init}_TSP_{sub_lr}_{acc_weight}_{vel_weight}'

if SNR == '--SNR':
    test_name += '_SNR_flat_0.01'

# train
os.system(f'python3 train.py --test-name={test_name} --n-shots={n_shots}'
         f' --trajectory-learning={trajectory_learning} --sub-lr={sub_lr}  --initialization={init} '
         f'--batch-size={batch_size}  --lr={rec_lr} --num-epochs={num_epochs} --acc-weight={acc_weight} '
         f'--vel-weight={vel_weight} --data-path={data_path} --sample-rate={sample_rate}'
         f' --data-parallel --interp_gap={interp_gap} {TSP} {SNR}')

# reconstruct and eval
os.system(f'python3 reconstructe_nosave.py --test-name={test_name} --data-split=val '
          f'--batch-size={batch_size} --data-path={data_path} --sample-rate={sample_rate}')

# reconstructe and save all the slices
# os.system(f'CUDA_VISIBLE_DEVICES={device} python3 reconstructe.py --test-name={test_name} --data-split=val '
#           f'--batch-size={batch_size} --data-path={data_path} --sample-rate={sample_rate}')
# eval all saved slices
# os.system(f'python3 common/evaluate.py --test-name={test_name} '
#           f'--data-path={data_path}')