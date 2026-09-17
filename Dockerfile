FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app/echosight

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R app:app /home/app/echosight

USER app

CMD ["python", "-m", "src.app"]
