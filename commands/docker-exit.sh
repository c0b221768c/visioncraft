#!/bin/bash

echo "Stop container && Remove"
docker compose -f docker-compose.yml down

echo "Done!"