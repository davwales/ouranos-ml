docker build -t talos-ml .
docker rm talos-ml
docker run --gpus=all -d -p 8000:8000 --name talos-ml talos-ml
docker image prune -f
