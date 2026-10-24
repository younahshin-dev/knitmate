FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python build_index.py || true
EXPOSE 8501 8000
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
