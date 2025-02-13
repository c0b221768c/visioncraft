#!/bin/bash

echo "added to access"
xhost +local:

echo "Connect container"
docker exec -it visioncraft-visioncraft-1 bash