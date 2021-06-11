xhost +local:docker
set XSOCK /tmp/.X11-unix
set XAUTH /tmp/.docker.xauth 
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -