# GPU Test

A basic training run using HuggingFace to stress a GPU and test it.

## Dependencies

This needs Python along with a few HuggingFace libraries as below

```
pip install transformers[torch] datasets
```

## Docker image creation

The Docker image was created using the Dockerfile and these commands to push it to DockerHub

```
docker build -t jakelever/gputest .
docker push jakelever/gputest
```
