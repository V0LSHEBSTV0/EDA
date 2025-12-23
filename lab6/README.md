Как запустить?


cd в lab6

docker build -t shoppers-model:clean .

docker run -p 5000:5000 -p 9050:9050 shoppers-model:clean

потом http://127.0.0.1:9050 для дашборда
и http://127.0.0.1:5000 для api модели
