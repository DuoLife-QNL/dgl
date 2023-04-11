import os

lower_bound = 0.001
step = 0.001
total_its = 10

file_path = '/home/lihz/Codes/dgl/MyCodes/GNNAutoScale/examples/train_gcn.py'
log_path_root = '/home/lihz/Codes/dgl/MyCodes/logs/GNNAutoSacle-DGL/GCN/Citeseer3'
prof_path_root = '/home/lihz/Codes/dgl/MyCodes/Profiling/GNNAutoScale/GCN/Citeseer3'
run_dir = '/home/lihz/Codes/dgl/MyCodes'

os.chdir(run_dir)
if not os.path.exists(log_path_root):
    os.makedirs(log_path_root)
if not os.path.exists(prof_path_root):
    os.makedirs(prof_path_root)

for i in range(0, total_its):
    lr = lower_bound + step * i
    prof_name = 'inference_with_hr_lr_{:.3f}'.format(lr)
    prof_path = os.path.join(prof_path_root, prof_name)
    command = 'python {} --lr={} --prof-path={}'.format(file_path, lr, prof_path)

    log_name = 'lr={:.3f}.log'.format(lr)
    log_path = os.path.join(log_path_root, log_name)

    os.system('{} > {}'.format(command, log_path))
