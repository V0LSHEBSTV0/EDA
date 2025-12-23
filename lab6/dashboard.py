from explainerdashboard import ExplainerDashboard

# Просто грузим то, что сгенерировали при сборке
db = ExplainerDashboard.from_config("dashboard.yaml")
db.run(host='0.0.0.0', port=9050)
