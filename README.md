
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

### Última Ejecución Completa del Pipeline

- **Fecha y Hora:** 2024-12-09 16:21:53  

---

## **Instalación y Ejecución del Pipeline**

1. **Clonado del Repositorio**  
Clona el repositorio usando `git`:

```bash
git clone https://github.com/MarioPasc/project_template
cd project_template
```

2. **Instalación de Dependencias**  
Ejecuta el script de configuración para instalar las dependencias:

```bash
./code/setup.sh
```

3. **Ejecución del Pipeline**  
Lanza el pipeline completo:

```bash
./code/launch.sh
```

---

## **Parámetros de Ejecución**

En el script **`launch.sh`**, se encuentran las siguientes variables configurables:

| **Variable**  | **Valor por Defecto** | **Descripción**                                                                                                                                                          |
|---------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OPTIMIZE**  | `false`               | Controla si se realiza la optimización bayesiana de hiperparámetros. Si está en `true`, se retomará desde el último intento si existen archivos `.db` en `results/`.     |
| **TRIALS**    | `150`                 | Define el número de intentos del algoritmo de optimización bayesiana. Se recomienda **100-150 iteraciones** para replicar los resultados del estudio original.           |


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
    │   └── launch.sh*                                        # Script principal para ejecutar el pipeline
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