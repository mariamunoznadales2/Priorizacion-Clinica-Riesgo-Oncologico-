# Caso cáncer: IA clínica para priorización de riesgo oncológico

Estudio de viabilidad sobre un sistema de apoyo a la priorización clínica de pacientes con posible riesgo oncológico usando datos tabulares multimodales y modelos de Machine Learning.

El objetivo del proyecto no es diagnosticar cáncer automáticamente. El objetivo es evaluar si los datos disponibles permiten ordenar pacientes por riesgo, generar alertas revisables y apoyar decisiones clínicas bajo supervisión médica.

```text
Paciente -> datos disponibles -> modelo estima riesgo -> alerta si supera umbral -> revisión clínica -> decisión médica
```

## Resultado ejecutivo

El proyecto concluye que existe señal útil para priorización clínica, pero no suficiente para plantear un diagnóstico automático.

Modelo recomendado:

```text
HistGradientBoosting calibrado
```

Motivos principales:

- rendimiento comparable o ligeramente superior a la MLP,
- buen comportamiento en datos tabulares,
- menor complejidad que una red neuronal,
- probabilidades calibradas,
- umbral ajustable según política clínica,
- mejor defendibilidad ante un uso sanitario prudente.

## Entregables finales

| Entregable | Archivo | Propósito |
|---|---|---|
| Presentación final | `PRESENTACIÓN.pptx` | Estudio de viabilidad en formato exposición |
| Dashboard ejecutivo-operativo | `app.py` | Simulación visual de políticas clínicas, impacto, carga y riesgos |
| Notebook principal | `modelo.ipynb` | Preparación, modelos ML, MLP y evaluación final |
| Evaluación clínica | `evaluacion_clinica.ipynb` | Calibración, umbrales, leakage, fairness y error analysis |
| Validación avanzada | `validacion_clinica.ipynb` | DCA, bootstrap, stress test, ablation y flujo operativo |
| Informe técnico | `docs/TÉCNICO.md` | Desarrollo metodológico completo |
| Defensa oral | `docs/DEFENSA.md` | Narrativa, preguntas difíciles y mensajes clave |
| Propuesta empresarial | `docs/propuesta.md` | Valor hospitalario, business case, validación y marco ético |
| Metadatos | `metadata.md` | Descripción de variables y lógica generativa del dataset |
| Datos | `data/` | CSV usados por notebooks y dashboard |

## Datos del proyecto

Dataset final unido por `paciente_id`:

| Elemento | Valor |
|---|---:|
| Pacientes totales | 50.001 |
| Columnas iniciales | 38 |
| Variables predictoras válidas | 30 |
| Features tras preprocesamiento | 45 |
| Positivos `cancer = 1` | 9.644 |
| Prevalencia | 19,29% |
| Train | 40.000 pacientes |
| Test | 10.001 pacientes |

Fuentes integradas:

| Colección | Contenido |
|---|---|
| `bioquimicos.csv` | Analíticas sanguíneas |
| `clinicos.csv` | Diagnósticos y comorbilidades |
| `geneticos.csv` | Mutaciones oncogénicas |
| `economicos.csv` | Costes y uso de recursos sanitarios |
| `generales.csv` | Hábitos y variables generales |
| `sociodemograficos.csv` | Perfil social y demográfico |

## Decisión sobre variables

Variables excluidas por fuga de datos, consecuencia post-diagnóstico o ausencia de información:

```text
paciente_id
cancer
coste_total
coste_farmaco
num_ingresos
dias_hospital
vive
alcohol
```

Justificación:

- `coste_total`, `coste_farmaco`, `num_ingresos` y `dias_hospital` son variables asistenciales que pueden ocurrir después del diagnóstico o tratamiento.
- `vive` es una consecuencia clínica posterior y no debe usarse para cribado previo.
- `alcohol` es constante en el dataset y no aporta señal predictiva.
- `paciente_id` es identificador.
- `cancer` es la variable objetivo.

La comprobación empírica de leakage muestra el riesgo metodológico:

| Escenario | Features | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sin leakage | 30 | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 |
| Con variables post-diagnóstico | 36 | 0,65 | 0,990 | 0,970 | 0,980 | 0,999 | 0,993 |

El rendimiento casi perfecto con leakage no es clínicamente válido. El modelo defendible es el que elimina esas variables aunque las métricas sean más realistas.

## Resultados principales

Comparación final entre el mejor modelo ML y la MLP:

| Modelo | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 |
| MLP | 0,32 | 0,522 | 0,630 | 0,571 | 0,840 | 0,817 |

La MLP obtiene algo más de recall, pero HistGradientBoosting queda como opción más práctica por F1, AUC, precisión, simplicidad, calibración y explicabilidad operativa.

Calibración:

| Modelo | Brier score | AUC-ROC |
|---|---:|---:|
| HistGradientBoosting calibrado | 0,110 | 0,844 |
| MLP | 0,154 | 0,839 |
| HistGradientBoosting sin calibrar | 0,156 | 0,843 |

La calibración es importante porque el sistema comunica riesgo, no solo una etiqueta binaria.

## Políticas clínicas simuladas

Con el modelo calibrado se proponen tres políticas:

| Política | Threshold | Precisión | Recall | F1 | Alertas/1.000 | Detectados/1.000 | FP/1.000 | FN/1.000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cribado | 0,13 | 0,363 | 0,857 | 0,510 | 456 | 165 | 290 | 28 |
| Priorización | 0,34 | 0,557 | 0,605 | 0,580 | 210 | 117 | 93 | 76 |
| Alta precisión | 0,52 | 0,700 | 0,363 | 0,478 | 100 aprox. | 70 | 30 | 123 |

La política recomendada para un piloto inicial es `Priorización`, porque equilibra utilidad clínica y carga asistencial.

## Dashboard operativo

El dashboard de Streamlit convierte los resultados técnicos en una herramienta ejecutiva:

- selector de política clínica,
- impacto por 1.000 pacientes,
- pacientes priorizados,
- evidencia del modelo,
- recomendación ejecutiva,
- carga asistencial,
- coste operativo,
- fairness por subgrupos,
- riesgos y limitaciones.

Ejecutar:

```bash
source venv/bin/activate
streamlit run app.py
```

Si se parte de cero:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Ejecución de notebooks

Para reproducir el análisis completo:

```bash
source venv/bin/activate
jupyter notebook
```

Orden recomendado:

1. `modelo.ipynb`
2. `evaluacion_clinica.ipynb`
3. `validacion_clinica.ipynb`

Dependencias principales para el análisis:

```bash
pip install pandas numpy scikit-learn matplotlib keras torch jupyter shap
```

Si se quiere reconectar con base de datos:

```bash
pip install sqlalchemy pyodbc
```

## Propuesta de valor clínica y empresarial

El proyecto se formula como un sistema de apoyo a:

- cribado,
- priorización clínica,
- alerta temprana,
- revisión médica posterior,
- gestión de carga asistencial.

Compradores o usuarios posibles:

- hospitales,
- redes de atención primaria,
- unidades oncológicas,
- aseguradoras sanitarias,
- equipos de gestión clínica.

Dolores que aborda:

- listas de espera sin priorización,
- retrasos diagnósticos,
- uso ineficiente de especialistas,
- exceso de pruebas sin criterio de riesgo,
- falta de trazabilidad en la priorización.

KPIs relevantes:

- tiempo hasta revisión,
- alertas por 1.000 pacientes,
- alertas por profesional/día,
- coste por caso detectado,
- falsos negativos evitables,
- recall por subgrupo,
- porcentaje de alertas revisadas en plazo.

## Validación necesaria antes de uso real

El dataset es sintético. Por tanto, el proyecto demuestra viabilidad metodológica, no validez clínica definitiva.

Roadmap recomendado:

1. Validación retrospectiva en cohorte real.
2. Validación temporal con fechas reales.
3. Validación externa en otro centro.
4. Estudio prospectivo silencioso sin intervenir decisiones.
5. Piloto clínico supervisado.
6. Despliegue limitado con monitorización.

## Marco ético y regulatorio

Antes de producción real serían necesarios:

- base legal y cumplimiento RGPD,
- minimización de datos,
- seudonimización,
- control de acceso,
- revisión específica del uso de datos genéticos,
- explicabilidad suficiente para clínicos,
- trazabilidad de cada alerta,
- protocolo de fallo seguro,
- auditoría continua de sesgo.

Clasificación funcional prudente:

```text
Software de apoyo a la decisión clínica para priorización, no diagnóstico autónomo.
```

## Limitaciones

- Dataset sintético.
- Falta validación externa.
- Falta validación temporal real.
- Las variables genéticas pueden no estar disponibles antes del diagnóstico en todos los contextos.
- El rendimiento depende de la prevalencia de la población.
- Existen falsos positivos y falsos negativos.
- Las variables socioeconómicas y genéticas requieren revisión ética.
- El umbral óptimo depende de capacidad asistencial, coste clínico y objetivo operativo.

## Conclusión final

El proyecto demuestra una viabilidad moderada para usar modelos tabulares como apoyo a la priorización clínica del riesgo oncológico.

La decisión final es:

```text
Implantar, como piloto supervisado, un HistGradientBoosting calibrado para priorización clínica.
No usarlo como diagnóstico automático.
Validar externamente antes de cualquier despliegue real.
```

El valor principal del proyecto no es obtener métricas perfectas. Es mostrar una cadena completa y prudente:

```text
datos -> modelo -> calibración -> umbral clínico -> impacto operativo -> riesgos -> decisión responsable
```
