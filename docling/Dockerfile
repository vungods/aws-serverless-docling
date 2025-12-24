# Use AWS Lambda Python 3.13 base image (Amazon Linux 2023 minimal)
FROM public.ecr.aws/lambda/python:3.13

# Set the working directory to the Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Install system dependencies required by docling/OpenCV/PIL
# libGL is needed for image processing, libglib for general support
RUN dnf install -y \
    mesa-libGL \
    mesa-libGLU \
    libglvnd-glx \
    libgomp \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Copy requirements file first
COPY docling/requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Lambda
ENV TORCH_HOME=/tmp/torch
ENV HF_HOME=/tmp/huggingface
ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

# Copy the rest of the application code
COPY docling/ ${LAMBDA_TASK_ROOT}/

# Set the entry point for the Lambda function
CMD ["lambda_function.lambda_handler"]