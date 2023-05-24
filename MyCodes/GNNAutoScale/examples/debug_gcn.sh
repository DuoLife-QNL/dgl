cd /home/lihz/Codes/dgl/MyCodes
python -m debugpy --listen 0.0.0.0:5828 --wait-for-client GNNAutoScale/examples/gcn.py training_method=gas _dataset=Cora run_env=multi_gpu _dataset.epochs=10