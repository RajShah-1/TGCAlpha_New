#!/bin/bash
sudo yum install gcc -y
sudo yum install gcc-c++ -y
sudo yum install openmpi-devel -y

# mkdir MPI4Py
# cd MPI4Py
# wget https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz
# tar zxvf mpich-3.3.2.tar.gz

# cd mpich-3.3.2
# ./configure --enable-shared --prefix=/usr/local/mpich --disable-fortran
# make
# sudo make install

sudo yum install python-pip -y
sudo env MPICC=/usr/lib64/openmpi/bin/mpicc pip install mpi4py
echo PATH=$PATH:/usr/lib64/openmpi/bin >> .bashrc
export PATH=$PATH:/usr/lib64/openmpi/bin
chmod 400 ~/.ssh/id_rsa

sudo pip install numpy
sudo pip install pandas

# Not required as of now
# sudo yum install python3 -y