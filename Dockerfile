# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Set environment variables if needed
ENV PYTHONUNBUFFERED=1

# Command to run your FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
