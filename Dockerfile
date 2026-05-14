# Step 1: Start with Python base image
FROM python:3.10-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy requirements first (for caching)
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project files
COPY . .

# Step 6: Tell Docker which port to open
EXPOSE 5000

# Step 7: Command to run when container starts
CMD ["python", "app.py"]
