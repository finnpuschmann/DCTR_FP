# DCTR_FP

All of the functions and methods are inside the DCTR.py Python-File.

Notebooks are only used to call these functions.

Dockerfile to get this to run on the GPU is also included.
Should only require the Nvidia Drivers to be installed, no CUDA or extra libraries.
Tested on Ubuntu 22.04.

To set up the Docker, in the Terminal run: 
```
cd path/to/Dockerfile
docker build ./ -t $DOCKER_NAME
```
where $DOCKER_NAME is any name you want to give the created docker container

To run the Docker:
```
docker run -it -v $(realpath '$HOME_PATH'):/tf/home -v $(realpath $DATA_PATH):/tf/data -p 8888:8888 --gpus all $DOCKER_NAME
```
where $HOME_PATH is the path that will be linked to the Dockers Home dir (/tf/home) this should have access to DCTR.py and the Notebooks.
and   $DATA_PATH is linked to Dockers Data dir (/tf/data), this is where the lhe (or converted .npz) files should be located.
