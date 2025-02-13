#!/bin/bash

echo "Remove the container && Rebuild && Reboot"
docker compose -f compose.yml down

echo "Build the container"
docker compose -f compose.yml up --build -d

echo "Done!"
docker ps