# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies for Tesseract, OpenCV, and Tkinter
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr-nep \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY fuzzyDocumentScanner.py /app/fuzzyDocumentScanner.py

# Set the working directory
WORKDIR /app

# Run the Python script
CMD ["python", "fuzzyDocumentScanner.py"]
