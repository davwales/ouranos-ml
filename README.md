# ouranos-ml

## Description

This project houses both the experimentation and hosting stages of Ouranos machine learning projects. These projects are used to facilitate intelligent features in the Ouranos Pantheon application, such as providing chat completions for the Hermes module.

## Prerequistites

- You have must have `uv` installed

## Usage

### Running the API locally

1. If you have not already, create a virtual environment.

```
uv venv
```

2. Install dependencies.

```
uv sync
```

3. Run using uvicorn. Either of the following commands will work.

```
uv run start
uv run ouranos_ml
```

4. Run using docker.

```
docker build -t ouranos-ml .
docker run --gpus=all -d -p 8000:8000 ouranos-ml
```

### Running the latest image

You can pull the latest image from the repository and run it using the following command.

```
docker login registry.gitlab.com
docker pull registry.gitlab.com/talos8645929/ouranos-ml:latest
docker run --gpus=all -d -p 8000:8000 --name ouranos-ml -e LMSTUDIO_BASE_URL=host.docker.internal:1234 registry.gitlab.com/talos8645929/ouranos-ml:latest
```

### Running an experiment

1. Create your experiment within the `src/experiments` directory.
2. Add your experiment to the list of experiments in `src/experiments/main.py`.
3. Execute your experiment.

```
uv run experiment {your_experiment_name}
```

## Contributing

At this point in time I am not interested in having additional contributors.
