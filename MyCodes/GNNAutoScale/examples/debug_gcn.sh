cd /home/lihz/Codes/dgl/MyCodes
python -m debugpy --listen 0.0.0.0:5828 --wait-for-client GNNAutoScale/examples/gcn.py _dataset.epochs=10 training_method=graphfm _dataset=Reddit2