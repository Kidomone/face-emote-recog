Для создания:
docker run -d --name oracle-xe \
    -p 1521:1521 -p 5500:5500 \
    -e ORACLE_PASSWORD=admin \
    gvenzl/oracle-xe

docker start oracle-xe



Для логов:

docker logs -f oracle-xe

