# Usar una imagen base de Python
FROM python:3.12.0-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo el archivo requirements.txt primero
COPY requirements.txt .

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos Python (.py) al contenedor


COPY api.py api.py
COPY train.py train.py
COPY predict.py predict.py


# Exponer el puerto que usará Uvicorn
EXPOSE 8000

# Comando para iniciar la API usando Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--reload"]




