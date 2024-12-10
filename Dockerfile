# Base de la imagen
FROM python:3.10-slim

# Configuración del entorno
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHON_LIB=/app/Py_libs \
    PYTHONPATH="/app/code:/app/Py_libs"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    wget \
    gawk \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    cm-super && \
    apt-get clean

# Crear directorio para bibliotecas locales
RUN mkdir -p $PYTHON_LIB

# Copiar el código y scripts
COPY . /app

# Instalar dependencias de Python localmente
RUN pip install --target=$PYTHON_LIB numpy pandas stringdb igraph matplotlib cairocffi plotly \
    jinja2 networkx optuna requests psutil scienceplots kaleido gseapy seaborn UpSetPlot matplotlib-venn --upgrade

# Permitir ejecución de scripts
RUN chmod +x /app/code/execute.sh

# Ejecutar el pipeline
CMD ["/app/code/execute.sh"]
