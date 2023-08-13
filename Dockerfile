# Usa la imagen oficial de Python como base
FROM python:3.8

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de la aplicación al contenedor
COPY app.py .
COPY data.json .

# Instala las dependencias
RUN pip install Flask requests numpy pandas scikit-learn redis Flask-Cors

# Expone el puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]
