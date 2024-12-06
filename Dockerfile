# Use a lightweight Python base image
FROM python:3.12

# Install system dependencies for Tesseract, OpenCV, and Tkinter
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-nep \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the Python dependencies file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Command to run the application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "wsgi:app"]
# here -w 4 means that Gunicorn will spawn 4 worker processes
# and -b 0.0.0.0:8000 specifies the IP address and port that Gunicorn will listen on