# Mushroom AI Classificator

Este proyecto proporciona una API para predecir el tipo de hongo a partir de diferentes características usando un modelo XGBoost. La API está desplegada en un servicio gratuito y se puede acceder a través de Swagger.

## API desplegada en Render

La API está disponible en [https://mushroom-ai-classificator.onrender.com/docs](https://mushroom-ai-classificator.onrender.com/docs). 

**Importante:** La API puede tardar hasta **5 minutos** en estar operativa debido a que está hospedada en un servicio gratuito. 

### Endpoints principales

1. **Entrenar modelo** (`POST /train`): 
   - Carga un archivo `train.csv` para entrenar un modelo XGBoost.
   - El modelo entrenado estará disponible para su descarga.

2. **Realizar predicciones** (`POST /predict`): 
   - Carga el archivo `test.csv` y el modelo entrenado.
   - Obtén un archivo CSV con las predicciones.

## Instrucciones para ejecutar la API en Docker (localmente)

### Requisitos

- Docker instalado. Si no lo tienes, instálalo desde [Docker](https://www.docker.com/).

### Construcción y ejecución

1. **Clona el repositorio:**

   ```bash    
   git clone https://github.com/tuusuario/Mushroom-AI-Classificator.git
   cd Mushroom-AI-Classificator/Fase-3
   ```
2. **docker build -t mushroom-ai-classificator**
   ```bash
   docker build -t mushroom-ai-classificator .
   ```
3. **Ejecutar la imagen Docker**
  ```bash
  docker run -p 5000:5000 mushroom-ai-classificator
  ```
4. La api estara disponible en [http://localhost:5000/docs](http://localhost:5000/docs)

### Endpoints principales

1. **Entrenar modelo** (`POST /train`): 
   - Carga un archivo `train.csv` para entrenar un modelo XGBoost.
   - El modelo entrenado estará disponible para su descarga.

2. **Realizar predicciones** (`POST /predict`): 
   - Carga el archivo `test.csv` y el modelo entrenado.
   - Obtén un archivo CSV con las predicciones.

