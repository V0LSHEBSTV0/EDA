from explainerdashboard import ExplainerDashboard

# Загружаем из конфига
db = ExplainerDashboard.from_config("dashboard.yaml")

# Запускаем (порт 9050 стандартный для дашборда)
db.run(host='0.0.0.0', port=9050, use_waitress=True)
