# Deep-Learning-Network-2D-Natural-Image-Decomposition
Primitive Decomposition: Deep Learning Network 2D Natural Image Decomposition
Easy conda install from requirements.txt
```sh
conda create -n vda
conda activate vda
conda config --env --add channels conda-forge
conda install python=3 -y
conda install $(cat requirements.txt) -y
```


For dataset download

```sh
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
mkdir dataset
mv annotations.tar.gz dataset
mv images.tar.gz dataset
cd dataset
tar -xvzf annotations.tar.gz 
tar -xvzf images.tar.gz
rm annotations.tar.gz
rm images.tar.gz
```