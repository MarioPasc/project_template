
# **Explorando la Demencia Frontotemporal mediante Biología de Sistemas**

Este repositorio contiene el código y recursos asociados al análisis presentado en el estudio:  
**"Explorando la Demencia Frontotemporal mediante Biología de Sistemas: Un Enfoque Integrado"**  

Autores:

- Mario Pascual González
- Ainhoa Nerea Santana Bastante
- Carmen Rodríguez González
- Gonzalo Mesas Aranda
- Ainhoa Pérez González.  

---

## **Configuración del Entorno**

Las pruebas fueron realizadas en:  

- **Sistema Operativo:** Ubuntu 22.04.5 LTS
- **Kernel:** Linux 6.8.0-49-generic  
- **Procesador:** 13th Gen Intel i7-13700KF (24) @ 5.300GHz  
- **GPU:** NVIDIA GeForce RTX 3060  
- **Arquitectura:** x86_64  

### Ejecución Completa del Pipeline

- Primera ejecución: 2024-11-22 11:40:01 
- Última Ejecución: 2024-12-09 16:21:53  

---

## **Instalación y Ejecución del Pipeline**

### **Forma Principal: Ejecución Usando Docker Hub**

La forma más sencilla y recomendada es ejecutar el pipeline utilizando la [imagen preconstruida de Docker en Docker Hub](https://hub.docker.com/r/mpascualg/biosist_ftd). El script `launch.sh` se encargará de descargar y ejecutar la imagen.

1. **Clonado del Repositorio**  
   Clonar el repositorio usando `git`:

   ```bash
   git clone https://github.com/MarioPasc/project_template
   cd project_template
   ```

2. **Ejecución del Pipeline**  
   Lanzar el pipeline completo ejecutando:

   ```bash
   ./code/launch.sh
   ```

   Este script:
   - Descarga la imagen preconstruida desde **Docker Hub**: `mariopasc/biosist_ftd:latest`.
   - Ejecuta el pipeline dentro de un contenedor Docker.
   - Monta las carpetas locales `results/` y `code/data/` para almacenar los resultados.

---

### **Formas Alternativas**

Si se prefiere no usar la imagen preconstruida de Docker Hub, puedes optar por las siguientes alternativas:

#### **1. Construir la Imagen Docker Localmente**

Si quieres construir la imagen tú mismo a partir del **Dockerfile**, sigue estos pasos:

1. **Construir la Imagen**  
   Ejecutar el siguiente comando en la raíz del proyecto:

   ```bash
   docker build -t biosist_ftd:latest .
   ```

2. **Ejecución del Pipeline con la Imagen Local**  
   Utilizar el script `launch.sh`, que ahora ejecutará la imagen localmente construida:

   ```bash
   ./code/launch.sh
   ```

---

#### **2. Ejecución Manual sin Docker (No recomendado)**

Si no se puede usar Docker, también se pueden instalar las dependencias y ejecutar el pipeline directamente en su máquina local:

1. **Instalación de Dependencias**  
   Ejecutar el script `setup.sh` para instalar todas las dependencias de Python localmente:

   ```bash
   ./code/setup.sh
   ```

2. **Ejecución del Pipeline**  
   Lanzar el pipeline ejecutando `execute.sh`:

   ```bash
   ./code/execute.sh
   ```

> [!CAUTION]
> Es muy probable que la generación del PDF no funcione con esta vía de ejecución, ya que esa parte del flujo ha sido programada ad hoc para Docker

---


| **Forma**                           | **Comando**                          | **Requisitos**                       |
|-------------------------------------|--------------------------------------|--------------------------------------|
| **Principal (Docker Hub)**          | `./code/launch.sh`                   | Docker instalado                     |
| **Alternativa 1 (Docker Local)**    | `docker build -t biosist_ftd .`<br>`./code/launch.sh` | Docker instalado                     |
| **Alternativa 2 (Ejecución Local)** | `./code/setup.sh`<br>`./code/execute.sh` | Python instalado (>=3.10), Latex local (figuras), sin Docker |

### **Flujo de Trabajo Interno**  

1. **`launch.sh`**: Ejecuta Docker y monta las carpetas necesarias.  
2. **`execute.sh` (dentro del contenedor)**: Se encarga de ejecutar el pipeline completo, incluyendo:  
   - Descarga de datos de HPO.  
   - Construcción y análisis de la red PPI.  
   - Optimización bayesiana (si se habilita).  
   - Generación de resultados y figuras.  

---

## **Parámetros de Ejecución**

En el script **`execute.sh`**, se encuentran las siguientes variables:

| **Variable**  | **Valor por Defecto** | **Descripción**                                                                                                                                                          |
|---------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OPTIMIZE**  | `true`               | Controla si se realiza la optimización bayesiana de hiperparámetros. Si está en `true`, se retomará desde el último intento si existen archivos `.db` en `results/`.     |
| **TRIALS**    | `100`                 | Define el número de intentos del algoritmo de optimización bayesiana. Se recomienda **80-100 iteraciones** para replicar los resultados del estudio original.           |

> [!CAUTION]
> Si se quiere modificar cualquiera de estas variables, se deben especificar en la llamada a `launch.sh`, ya que Docker se encargará de pasarlas a `execute.sh`:
> ```bash
> OPTIMIZE=true TRIALS=100 ./code/launch.sh
> ```

> [!CAUTION]
> Si se fija `OPTIMIZE=false`, entonces se debe renombrar la carpeta `original_results` a `results`, para que tome los resultados guardados ahí. Esto se puede hacer fácilmente con:
> ```bash
> mv ./original_results ./results
> ```

> [!IMPORTANT]
> Si `results/` está vacío y `OPTIMIZE=true`, el ajuste se iniciará desde cero. Si los archivos `.db` existen, se reanudará el ajuste.

> [!NOTE]
> Se estima 1 minuto por intento para cada algoritmo de clustering (Leiden y Louvain). Este tiempo puede estar relacionado con la configuración de hardware, pero sobre todo depende del estado de la API.

---

## **Estructura del Proyecto**

La estructura del proyecto está organizada de la siguiente manera:

```yaml
📂 project_template/
    ├── 📂 code/                                              # Código fuente principal
    │   ├── 📂 clustering/                                    # Scripts de clustering
    │   ├── 📂 functional_analysis/                           # Scripts de análisis funcional
    │   ├── 📂 network/                                       # Scripts de construcción y análisis de redes
    │   ├── 📂 utils/                                         # Utilidades y funciones auxiliares
    │   ├── setup.sh*                                         # Script para instalación de dependencias
    │   ├── launch.sh*                                        # Ejecución de la imagen Docker
    │   └── execute.sh*                                       # Script principal para ejecutar el pipeline
    │
    ├── 📂 report/                                            # Documentación y reportes del estudio
    │   ├── ...                   
    │   └── report.pdf                                        # Artículo original del proyecto
    │
    ├── 📂 results/                                           # Resultados de las ejecuciones
    │      ├── clustering_results.json                        # Resultados de clustering
    │      ├── functional_analysis_leiden_max_enrichment.csv  # Resultados de enriquecimiento funcional sin filtrar
    │      ├── functional_analysis_leiden_max_modularity.csv  # Resultados de enriquecimiento funcional sin filtrar
    │      ├── filtered_results_leiden_max_enrichment.csv     # Resultados de enriquecimiento funcional filtrados
    │      ├── filtered_results_leiden_max_modularity.csv     # Resultados de enriquecimiento funcional filtrados
    │      ├── leiden_optimization.db                         # Resultados de optimización Leiden
    │      ├── multilevel_optimization.db                     # Resultados de optimización Louvain
    │      ├── networkAnalysisMetrics.tex                     # Métricas de análisis de la red
    │      ├── results_leiden.csv                             # Resultados de optimización Leiden (formato csv)
    │      ├── results_multilevel.csv                         # Resultados de optimización Louvain (formato csv)
    │      └── 📂 plots/                                      # Gráficos generados
    │              ├── 📂 clustering/                         # Visualización de clustering
    │              ├── 📂 functional_analysis/                # Gráficos de análisis funcional
    │              ├── 📂 network/                            # Gráficos de redes
    │              └── 📂 optimization/                       # Gráficos de optimización
    │
    ├── Dockerfile                                            # Dockerfile con la configuración del entorno Docker
    └── README.md                                             # Documentación principal del proyecto
```

---

## **Resultados**

Los resultados incluyen:  

1. **Clustering Funcional:** Generado con algoritmos **Leiden** y **Louvain**.  
2. **Análisis Funcional:** Rutas biológicas enriquecidas, incluyendo archivos `filtered_results` optimizados.  
3. **Optimización Bayesiana:** Bases de datos `.db` con resultados de hiperparámetros ajustados.  
4. **Visualización:** Gráficos detallados en la carpeta `results/plots/`.

## **Contacto**

Se deja como contacto al alumno coordinador del proyecto, Mario Pascual González:  
📧 [pascualgonzalez.mario@uma.es](mailto:pascualgonzalez.mario@uma.es)