docker cp hotel_bookings.csv namenode:hotel_bookings.csv
docker exec -it namenode bash
hdfs dfs -mkdir -p /data/hotel_data
hdfs dfs -D dfs.blocksize=67108864 -put hotel_bookings.csv /data/hotel_data/hotel_bookings.csv
exit

docker exec -it spark-worker-1 bash
apk add --update make automake gcc g++
apk add --update python-dev
apk add linux-headers
pip install numpy
exit

docker cp spark_app_optimized.py spark-master:spark_app.py
docker exec -it spark-master bash

apk add --update make automake gcc g++
apk add --update python-dev
apk add linux-headers
pip install findspark
pip install numpy

spark/bin/spark-submit spark_app.py
