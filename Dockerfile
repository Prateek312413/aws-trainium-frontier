# AWS Trainium2 Frontier Competition Container
FROM public.ecr.aws/neuron/pytorch-inference-neuronx:2.1.0-neuronx-py310-sdk2.16.0-ubuntu20.04

WORKDIR /workspace

# Install dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

ENV PYTHONUNBUFFERED=1

# Default speedrun command
CMD ["python", "train_speedrun.py", "--model-type", "base", "--duration-sec", "1800"]
