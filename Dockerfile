# Use Python 3.11 official slim image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --upgrade streamlit

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run Streamlit command by default
CMD ["streamlit", "run", "main.py"]
