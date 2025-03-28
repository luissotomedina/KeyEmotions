# Python image
FROM python:3.12-slim

# Install necessary packages for GUI applications and X11 forwarding
RUN apt-get update && apt-get install -y \
    python3-pyqt6 \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Actualize pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app

# Set environment for X11
ENV DISPLAY=:0

# Command to execute GUI
CMD ["python", "src/KeyEmotionsUI.py"]