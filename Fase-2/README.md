## Instrucciones para ejecutar la Fase 2

1. Ejectuar el comando `docker build -t mushroom-model ` en la carpeta Fase-2 para construir la imagen del contenedor.
2. Ejecutar el comando `docker run -it --rm --name mushroom-model mushroom-model` para correr el contenedor.
3. En la consola del contenedor puede ejecutar el comando `python Fase-2\train.py train.csv` para entrenar el modelo usando el archivo `train.csv` que se encuentra en la carpeta Fase-2 o especificar la ruta de un archivo diferente. El resultado del entrenamiento es un archivo `xgb_model + la fecha de creación del modelo.joblib` que se guarda en la carpeta del contenedor.
4. Para predecir una nueva instancia puede ejecutar el comando `python Fase-2\predict.py test.csv` en la consola del contenedor, el cual usa el archivo `test.csv` que se encuentra en la carpeta Fase-2 o especificar la ruta de un archivo diferente. El resultado de la predicción es un archivo `predictions.csv` que se guarda en la carpeta del contenedor y contiene las predicciones de las instancias del archivo dado, este archivo tiene dos columnas, la primera es el id de la instancia y la segunda es la predicción de la clase de la instancia.
5. Para salir del contenedor puede ejecutar el comando `exit` en la consola del contenedor.
