# GPU Test

A basic training run using HuggingFace to stress a GPU and test it.

## Dependencies

This needs Python along with a few HuggingFace libraries as below

```
pip install transformers[torch] datasets
```

## Running the Docker image

The Docker image is uploaded to DockerHub as [jakelever/gputest](https://hub.docker.com/repository/docker/jakelever/gputest). To run it with GPUs, use the command below

```
docker run --gpus all jakelever/gputest
```

## Docker image creation

The Docker image was created using the Dockerfile and these commands to push it to DockerHub

```
docker build -t jakelever/gputest .
docker push jakelever/gputest
```
