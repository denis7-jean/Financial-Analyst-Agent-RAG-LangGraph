# 1. Use Python 3.10-slim as the base image (lightweight and avoids Python 3.9 warnings)
FROM python:3.10-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker caching mechanisms
# (This prevents re-installing dependencies if only code changes)
COPY requirements.txt .

# 4. Install Python dependencies without caching to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code into the container
# (This includes app.py, the src folder, and your local ChromaDB vector store)
COPY . .

# 6. Expose port 8080
# (Google Cloud Run listens on port 8080 by default)
EXPOSE 8080

# 7. Command to run the application
# Streamlit must listen on 0.0.0.0 to be accessible externally
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]
