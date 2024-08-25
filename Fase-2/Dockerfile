# Usar una imagen base de Python
FROM python:3.12.0-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos requirements.txt y los archivos .py al contenedor
COPY requirements.txt requirements.txt
COPY train.py train.py
COPY predict.py predict.py
COPY test.csv test.csv
COPY train.csv train.csv

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Establecer el comando por defecto para iniciar una terminal bash
CMD ["bash"]
