cd /home/lihz/Codes/dgl/MyCodes/GNNAutoScale 
python ./setup.py install
cd /home/lihz/Codes/dgl
python -m debugpy --listen 0.0.0.0:5828 --wait-for-client ./MyCodes/GNNAutoScale/examples/train_gcn.py