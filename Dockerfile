
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY streamlit_dashboard/requirements.txt ./streamlit_dashboard/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
     -r streamlit_dashboard/requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", \
     "streamlit_dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
