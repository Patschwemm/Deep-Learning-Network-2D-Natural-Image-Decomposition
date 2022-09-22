# Deep-Learning-Network-2D-Natural-Image-Decomposition
Primitive Decomposition: Deep Learning Network 2D Natural Image Decomposition
Easy dependency install from requirements.txt
```sh
conda create -n cv
conda activate cv
pip install -r requirements.txt
```


Basic structure: \\

Backbone of the network (to be worked on): backbone_training.py

Decomposition Network and primitive modules: network.py

encoder part for feature extraction: encoder.py

losses for quasi unsupervised task: losses.py

primitive geometry and distance field computation: field_geometry.py

Different notebooks for different datasets (to be worked on to split into python files and do one general training file)