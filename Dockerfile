# Use the official Apache Spark image with Python support
FROM apache/spark-py:latest

# Prevent Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Force Python stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED=1

# Set the working directory in the Docker container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

USER root 

# Install Python dependencies as root
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and change ownership of the app folder
RUN adduser --uid 5678 --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Copy the rest of your application files into the container
COPY . /app

# Expose ports for Jupyter and Spark UI
EXPOSE 8888 4040

# Default command to run the Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]
