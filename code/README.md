# **Estructura del Código**

Esta carpeta contiene el código principal del proyecto, organizado en submódulos funcionales que corresponden a distintas etapas del análisis de biología de sistemas. Los scripts están estructurados para garantizar claridad, modularidad y reproducibilidad.

---

## **Estructura de Submódulos**

```yaml
📂 code/
    ├── 📂 clustering/           # Scripts relacionados con algoritmos de clustering
    ├── 📂 functional_analysis/  # Scripts para análisis de enriquecimiento funcional
    ├── 📂 network/              # Scripts para construcción y análisis de la red PPI
    ├── 📂 utils/                # Funciones y herramientas auxiliares
    ├── setup.sh*                # Script para instalación de dependencias
    ├── launch.sh*               # Ejecución de la imagen Docker
    └── execute.sh*              # Script principal para ejecutar el pipeline completo
```

---

### **Descripción de Submódulos**

1. **`clustering/`**  
   Contiene scripts que implementan algoritmos de clustering, como **Leiden** y **Louvain**, sobre la red de interacción proteína-proteína (PPI). Además, incluye la optimización bayesiana de hiperparámetros para maximizar la calidad del clustering.

2. **`functional_analysis/`**  
   Scripts dedicados al análisis de enriquecimiento funcional. Identifica procesos biológicos, funciones moleculares y rutas significativamente enriquecidas en los módulos de genes detectados.

3. **`network/`**  
   Contiene scripts para construir y analizar redes de interacción proteína-proteína (PPI) a partir de los genes asociados al fenotipo **FTD**.

4. **`utils/`**  
   Este submódulo proporciona funciones auxiliares y herramientas comunes utilizadas en otros submódulos. Incluye la descarga de la red, cambio de formato `.tsv` a iGraph, o carga de ficheros `.json`.

---

## **Scripts Principales**

- **`setup.sh`**  
   Instala todas las dependencias necesarias del proyecto.

- **`execute.sh`**  
   Script principal para ejecutar el pipeline completo. Gestiona todas las etapas: construcción de la red, clustering, optimización y análisis funcional.

- **`launch.sh`**  
   Script principal para la ejecución de la DockerImage.

>[!NOTE] 
> **Construcción de la imagen local**. Se deja documentado el proceso:
> ```yaml
> docker build -t biosist_ftd:latest .
> docker images # Comprobar que se ha creado
> docker tag biosist_ftd:latest mpascualg/biosist_ftd:latest # Nombre de usuario
> docker push mpascualg/biosist_ftd:latest # Push a Docker Hub
> ```
> Para hacer pruebas locales, se puede ejecutar el docker durante un momento:
> ```yaml
> OPTIMIZE=true TRIALS=5 ./code/launch.sh
> ...
> docker ps # Mirar el ID de ejecución
> docker stop {ID}