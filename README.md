# ouranos-ml

## Description
This project houses both the experimentation and hosting stages of Ouranos machine learning projects. These projects are used to facilitate intelligent features in the Ouranos Pantheon application, such as providing chat completions for the Hermes module.

## Usage
If you wish to run the API you can do so by following the following steps.

1. If you have not already, created a virtual environment.
    1. `python -m venv ./env`
2. Install dependencies.
    1. `pip install -r requirements.txt
3. Run using uvicorn.
    1. `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
4. Run using docker.
    1. `docker build -t ouranos-ml .`
    2. `docker run --gpus=all -d -p 8000:8000 ouranos-ml`

If you wish to pull the latest container built for the repository, you can follow the following steps.

1. Authenticate with the docker repository.
    1. `docker login registry.gitlab.com`
2. Pull the image.
    1. `docker pull registry.gitlab.com/talos8645929/ouranos-ml:latest`
3. Run the image.
    1. `docker run --gpus=all -d -p 8000:8000 --name ouranos-ml registry.gitlab.com/talos8645929/ouranos-ml:latest`

## Contributing
At this point in time I am not interested in having additional contributors. 
