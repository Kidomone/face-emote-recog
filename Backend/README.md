### Создание Docker-контейнера
```
docker run -d --name oracle-xe \
    -p 1521:1521 -p 5500:5500 \
    -e ORACLE_PASSWORD=admin \
    gvenzl/oracle-xe
```

### Запуск контейнера
```
docker start oracle-xe
```

### Просмотр логов
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