To check if the current project is using CUDA, 
Run:

uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
