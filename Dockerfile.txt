# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies (optional, modify based on your needs)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py"]
