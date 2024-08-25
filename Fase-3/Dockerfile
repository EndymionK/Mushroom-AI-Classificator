# Usar una imagen base de Python
FROM python:3.12.0-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos requirements.txt y los archivos .py al contenedor
COPY requirements.txt requirements.txt
COPY train.py train.py
COPY predict.py predict.py


# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Exponer el puerto que usará Uvicorn
EXPOSE 8000

# Comando para iniciar la API usando Uvicorn
CMD ["uvicorn", "api:app", "--reload"]

