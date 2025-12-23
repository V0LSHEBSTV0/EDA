FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 1. Обучаем модель
RUN python train_model.py

# 2. Генерируем "тяжелый" дашборд заранее (сохранит dashboard.yaml)
RUN python generate_dashboard.py

# 3. Подготовка скрипта запуска
RUN chmod +x run.sh
EXPOSE 5000 9050
CMD ["./run.sh"]
