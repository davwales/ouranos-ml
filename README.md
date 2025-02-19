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

## Contributing
At this point in time I am not interested in having additional contributors. 
