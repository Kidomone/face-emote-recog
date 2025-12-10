### Создание Docker-контейнера
```
docker run -d --name oracle-xe \
    -p 1521:1521 -p 5500:5500 \
    -e ORACLE_PASSWORD=admin \
    gvenzl/oracle-xe

docker start oracle-xe
```

### Логирование
```
docker logs -f oracle-xe
```

### Запуск сервера
```
cd .venv/Scripts
activate
cd ../..

uvicorn backend.main:app --reload
```