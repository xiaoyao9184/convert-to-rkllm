# Convert to RKLLM

A Gradio Docker image built via GitHub Actions.

## Why

This project is similar to [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) `rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export`.

It uses GitHub Actions to build and publish Docker images, and to sync with Hugging Face Gradio Spaces.  
The goal is to keep the entire process clean and minimal, without custom configuration files.

## Spaces

The Hugging Face Space for this project is located at:  
👉 [xiaoyao9184/convert-to-rkllm](https://huggingface.co/spaces/xiaoyao9184/convert-to-rkllm)

## Tags

The Docker images are published to Docker Hub under:  
👉 [xiaoyao9184/convert-to-rkllm](https://hub.docker.com/r/xiaoyao9184/convert-to-rkllm)

Image tags are generated using the `commit_id` and branch name (`main`).  
See the tagging workflow in [docker-image-tag-commit.yml](./.github/workflows/docker-image-tag-commit.yml).

> **Note:** Currently, only the `linux/amd64` platform is supported.

## Change / Customize

You can fork this project and build your own image.  
You will need to provide the following secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HF_USERNAME`, and `HF_TOKEN`.

See [docker/login-action](https://github.com/docker/login-action#docker-hub) for more details.
