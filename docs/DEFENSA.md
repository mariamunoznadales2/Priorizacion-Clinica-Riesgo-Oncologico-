# Presentación final: sistema de apoyo al cribado y priorización de cáncer
## Guía extendida para defensa del proyecto

Este documento complementa al [`TÉCNICO.md`](TÉCNICO.md). El informe técnico desarrolla la metodología completa; esta guía está pensada para preparar la presentación final, ordenar la narrativa y tener respuestas claras ante preguntas metodológicas, clínicas y éticas.

La idea central de la presentación debe ser sencilla:

```text
No se ha construido un diagnóstico automático.
Se ha evaluado si datos tabulares de pacientes permiten priorizar revisión clínica.
```

El proyecto no debe venderse como un sistema que sustituye al médico. Debe defenderse como un sistema de apoyo que calcula riesgo, genera alertas y ayuda a decidir qué pacientes revisar antes.

## Resumen ejecutivo

El proyecto analiza un dataset tabular multimodal de pacientes para predecir la variable `cancer`. Se integran datos bioquímicos, clínicos, genéticos, económicos, generales y sociodemográficos. Tras limpiar variables con fuga de información, se comparan modelos clásicos de Machine Learning con una red neuronal multicapa.

El resultado final recomienda:

```text
HistGradientBoosting calibrado
```

Esta recomendación no se basa solo en una métrica. Se apoya en:

- rendimiento en test,
- F1 ligeramente superior,
- AUC estable,
- mejor calibración tras `CalibratedClassifierCV`,
- menor complejidad que la MLP,
- comportamiento robusto en validaciones adicionales,
- posibilidad de elegir thresholds según objetivo clínico.

La conclusión clínica es prudente:

```text
El sistema puede ser útil como apoyo al cribado o priorización.
No es válido como diagnóstico automático.
```

## Mensaje principal para la defensa

Si hubiera que explicar el proyecto en menos de un minuto:

> Este trabajo evalúa si un sistema de Machine Learning puede ayudar a priorizar pacientes con posible cáncer usando datos tabulares disponibles antes del diagnóstico. Se comparan modelos clásicos y una MLP, se eliminan variables con fuga de datos, se optimizan umbrales de forma simétrica y se traduce el resultado a un flujo clínico realista. El modelo recomendado es HistGradientBoosting calibrado, con AUC alrededor de 0,84 y F1 alrededor de 0,58. La herramienta no diagnostica: genera alertas para revisión médica posterior.

Esta frase resume lo más importante:

- objetivo realista,
- metodología limpia,
- resultado cuantificado,
- prudencia clínica,
- utilidad operativa.

## Qué problema resuelve

En un entorno clínico real, no todos los pacientes pueden revisarse con la misma prioridad. Un sistema de riesgo puede ayudar a ordenar casos y detectar perfiles que merecen revisión adicional.

El problema no es simplemente clasificar pacientes. El problema práctico es:

```text
Dado un conjunto de pacientes, identificar cuáles deberían priorizarse para revisión clínica.
```

Por eso las métricas relevantes no son solo accuracy o AUC. Importan especialmente:

- recall de la clase positiva,
- precisión de las alertas,
- F1,
- calibración de probabilidades,
- falsos negativos,
- carga asistencial por cada 1.000 pacientes,
- variabilidad por subgrupos,
- dependencia de prevalencia.

## Qué NO resuelve

Es importante decir explícitamente lo que el sistema no hace.

El sistema no:

- confirma cáncer,
- descarta cáncer,
- sustituye pruebas diagnósticas,
- sustituye criterio médico,
- decide tratamientos,
- debe desplegarse sin validación externa.

Esta aclaración no debilita el proyecto. Al contrario: lo hace más serio y más parecido a un sistema clínico real.

## Estructura del proyecto

El proyecto queda dividido en tres notebooks:

| Notebook | Papel en el proyecto |
|---|---|
| `modelo.ipynb` | Construcción del dataset, preparación, modelos baseline, MLP, ajuste de umbrales y evaluación final |
| `evaluacion_clinica.ipynb` | Traducción clínica: umbrales, calibración, leakage, fairness, falsos negativos, coste-beneficio y policy card |
| `validacion_clinica.ipynb` | Validación avanzada: DCA, auditoría interseccional, bootstrap, prevalencia, ablation e implementación clínica |

El README principal resume los resultados. Este documento sirve como apoyo para convertir esos resultados en presentación.

## Datos utilizados

El dataset final combina seis fuentes:

| Fuente | Contenido |
|---|---|
| Bioquímicos | Analíticas y medidas biológicas |
| Clínicos | Antecedentes y comorbilidades |
| Genéticos | Mutaciones relevantes |
| Económicos | Costes y variables asistenciales |
| Generales | Hábitos y variables generales |
| Sociodemográficos | Edad, ingresos, educación, zona, estado civil |

Dimensión final:

```text
50.001 pacientes
38 columnas antes de exclusiones
30 variables predictoras válidas tras eliminar leakage
45 features tras preprocesamiento
```

Variable objetivo:

```text
cancer
```

Distribución:

| Dataset | Pacientes | Positivos | Prevalencia |
|---|---:|---:|---:|
| Total | 50.001 | 9.644 | 19,29% |
| Train | 40.000 | 7.715 | 19,29% |
| Test | 10.001 | 1.929 | 19,29% |

El problema está desbalanceado. Por eso una accuracy alta no basta: un modelo podría acertar muchos negativos y aun así ser malo detectando cáncer.

## Decisión metodológica crítica: fuga de datos

Una parte esencial del proyecto fue detectar y retirar variables que no serían válidas para predicción previa.

Variables excluidas por posible fuga o por no aportar información:

```text
coste_total
coste_farmaco
num_ingresos
dias_hospital
vive
alcohol
```

También se excluyen:

```text
paciente_id
cancer
```

La razón es que las variables post-diagnóstico pueden estar disponibles después del diagnóstico o ser consecuencia del proceso asistencial. Además, `alcohol` se excluye porque el metadata actualizado indica que es constante. Incluir estas variables permitiría predecir muy bien en el dataset, pero no serviría para un sistema real de cribado.

Prueba empírica:

| Escenario | Features | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|
| Sin leakage | 30 | 0,576 | 0,843 | 0,829 |
| Con variables post-diagnóstico | 36 | 0,980 | 0,999 | 0,993 |

Lectura para la presentación:

```text
Cuando se permite leakage, el rendimiento se dispara artificialmente.
Cuando se elimina, el rendimiento baja, pero el estudio se vuelve clínicamente defendible.
```

Esta es una de las partes más fuertes del proyecto, porque demuestra criterio metodológico.

## Preprocesamiento

El pipeline sigue una secuencia estándar:

1. Unión de tablas por `paciente_id`.
2. Definición de `X` e `y`.
3. Eliminación de identificador y variables con fuga.
4. Split estratificado train/test.
5. Escalado de variables numéricas.
6. One-hot encoding de categóricas.
7. Ajuste del preprocesador solo en train.

Variables categóricas:

```text
tipo_seguro
actividad_fisica
nivel_educativo
nivel_ingresos
zona
estado_civil
```

Punto importante para defensa:

```text
El preprocesamiento se ajusta en train y se aplica a test.
No se usa test para entrenar transformaciones ni elegir umbrales.
```

## Modelos evaluados

Se evaluaron modelos clásicos y una red neuronal.

Modelos clásicos:

| Modelo | Rol |
|---|---|
| Logistic Regression | Baseline lineal |
| Random Forest | Ensemble robusto |
| Extra Trees | Ensemble aleatorizado e interpretabilidad |
| HistGradientBoosting | Modelo tabular fuerte |

Red neuronal:

```text
MLP con Keras 3 + backend Torch
```

La MLP se diseñó con:

- capas densas de 128, 64 y 32 neuronas,
- Batch Normalization,
- ReLU,
- Dropout,
- salida sigmoide,
- `class_weight`,
- EarlyStopping,
- ReduceLROnPlateau.

Mensaje para presentación:

```text
No se comparó el modelo clásico contra una red débil.
La MLP tiene regularización, validación interna y ajuste de umbral.
```

## Resultados baseline

Con threshold estándar `0.50`, los modelos clásicos dieron:

| Modelo | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,441 | 0,754 | 0,557 | 0,843 | 0,768 |
| Logistic Regression | 0,435 | 0,759 | 0,553 | 0,841 | 0,763 |
| Extra Trees | 0,577 | 0,502 | 0,537 | 0,831 | 0,833 |
| Random Forest | 0,676 | 0,361 | 0,471 | 0,835 | 0,843 |

Lectura:

- Random Forest tiene buena accuracy, pero recall bajo.
- Logistic Regression y HistGradientBoosting tienen más recall.
- HistGradientBoosting tiene el mejor F1 inicial.
- En cribado, el recall importa más que la accuracy.

## Ajuste de umbral

El proyecto no se queda con `threshold = 0.50`. Esto es importante porque:

- las clases están desbalanceadas,
- el coste de FN y FP no es igual,
- distintos usos clínicos requieren distintas políticas.

Se barren thresholds de:

```text
0.10 a 0.90
```

La elección se hace sobre validación interna.

Corrección metodológica clave:

```text
Se optimiza el threshold de HistGradientBoosting y de MLP con la misma regla.
```

Esto evita comparar un modelo con threshold optimizado contra otro con threshold por defecto.

## Evaluación final

Resultados finales en test:

| Modelo | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 |
| MLP | 0,32 | 0,522 | 0,630 | 0,571 | 0,840 | 0,817 |

Interpretación correcta:

```text
HistGradientBoosting gana por F1, AUC, precisión y accuracy.
La MLP tiene mayor recall.
La diferencia es pequeña.
```

La conclusión no debe ser:

```text
El Boosting es claramente superior.
```

La conclusión debe ser:

```text
El Boosting es la opción operativa más defendible por rendimiento similar o ligeramente mejor, simplicidad e interpretabilidad.
```

## Calibración de probabilidades

Para uso clínico, no basta con ordenar pacientes. Si se informa un riesgo, la probabilidad debe ser razonablemente interpretable.

Se añadió:

```python
CalibratedClassifierCV(method="sigmoid", cv=5)
```

Resultado:

| Modelo | Brier score | AUC-ROC |
|---|---:|---:|
| HistGradientBoosting calibrado | 0,110 | 0,844 |
| MLP | 0,154 | 0,839 |
| HistGradientBoosting sin calibrar | 0,156 | 0,843 |

Métricas del Boosting calibrado:

| Modelo | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting calibrado | 0,29 | 0,512 | 0,664 | 0,578 | 0,844 | 0,813 |

Mensaje para la defensa:

```text
El modelo calibrado no solo clasifica: produce probabilidades más defendibles.
```

## Policy card operativa

La policy card traduce métricas técnicas a decisiones.

Modelo recomendado:

```text
HistGradientBoosting calibrado
```

Uso:

```text
alerta -> revisión médica -> prueba confirmatoria si procede
```

Políticas:

| Política | Threshold | Precisión | Recall | F1 | Detectados/1.000 | FP/1.000 | FN/1.000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cribado | 0,13 | 0,363 | 0,857 | 0,510 | 165,3 | 290,3 | 27,6 |
| Equilibrado | 0,34 | 0,557 | 0,605 | 0,580 | 116,8 | 92,8 | 76,1 |
| Alta precisión | 0,52 | 0,700 | 0,363 | 0,478 | 70,1 | 30,0 | 122,8 |

Cómo explicarlo:

- En cribado se acepta revisar más pacientes para reducir FN.
- En priorización se controla la carga asistencial.
- En alta precisión se generan menos alertas, pero se pierden más casos.

## Simulación hospitalaria

Con la política equilibrada:

| Resultado operativo | Pacientes por 1.000 |
|---|---:|
| Alertas generadas | 210 |
| Casos detectados | 117 |
| Falsos positivos revisados | 93 |
| Casos no alertados | 76 |

Lectura:

```text
Por cada 1.000 pacientes, el sistema generaría unas 210 alertas.
```

Esto permite hablar de carga asistencial:

- 210 pacientes serían priorizados,
- 117 corresponderían a casos positivos detectados,
- 93 serían falsas alarmas,
- 76 positivos no recibirían alerta.

Mensaje clave:

```text
La utilidad depende de que exista un circuito posterior de revisión.
```

## Error analysis: falsos negativos

Se analizaron los falsos negativos porque son el error clínicamente más importante.

Casos:

| Grupo | Casos |
|---|---:|
| Verdaderos positivos | 1161 |
| Falsos negativos | 768 |

Principales diferencias:

| Variable | Media TP | Media FN | Diferencia |
|---|---:|---:|---:|
| `trigliceridos` | 172,1 | 155,1 | -16,9 |
| `glucosa` | 112,2 | 101,9 | -10,3 |
| `colesterol` | 203,7 | 194,2 | -9,4 |
| `obesidad` | 0,659 | 0,380 | -0,279 |
| `fumador` | 0,682 | 0,440 | -0,242 |
| `mut_BRCA1` | 0,295 | 0,070 | -0,224 |

Interpretación:

```text
Los falsos negativos parecen tener señales menos extremas.
```

Es decir, el modelo detecta mejor perfiles con mayor carga metabólica, hábitos de riesgo o mutaciones relevantes. Los casos más discretos son más difíciles.

Conclusión:

```text
Para reducir FN de forma importante harían falta mejores señales clínicas.
```

Ejemplos:

- síntomas,
- antecedentes familiares,
- marcadores tumorales,
- imagen médica,
- analíticas longitudinales.

## Fairness y mitigación

Se evaluó si variables socioeconómicas podían afectar al rendimiento y se probó una mitigación:

```text
entrenar sin variables socioeconómicas
```

Variables retiradas:

```text
nivel_ingresos
tipo_seguro
nivel_educativo
zona
estado_civil
```

Resultado:

| Escenario | Features | F1 | AUC-ROC | Gap recall |
|---|---:|---:|---:|---:|
| Completo sin leakage | 30 | 0,576 | 0,843 | 0,089 |
| Sin socioeconómicas | 25 | 0,559 | 0,831 | 0,061 |

Lectura:

- baja el F1 en 0,017,
- baja el AUC en 0,012,
- mejora el gap de recall en 0,028.

Conclusión:

```text
Eliminar variables socioeconómicas reduce algo el rendimiento, pero mejora una dimensión de equidad.
```

Esto permite defender dos opciones:

- modelo completo con auditoría continua,
- modelo sin socioeconómicas si se prioriza prudencia ética.

## Validación prospectiva simulada

Como no hay fechas reales, se simula:

```text
primer 80% por paciente_id -> pasado
último 20% por paciente_id -> futuro
```

Resultado:

| Escenario | Train pasado | Test futuro | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prospective split simulado | 40000 | 10001 | 0,61 | 0,509 | 0,642 | 0,568 | 0,841 | 0,816 |

Lectura:

```text
No aparece una caída fuerte al evaluar en el bloque futuro simulado.
```

Limitación:

```text
No sustituye una validación temporal real.
```

## Decision Curve Analysis

La DCA evalúa beneficio neto frente a:

- revisar a todos,
- no revisar a nadie.

Resultado:

| Modelo | Threshold mínimo con beneficio | Threshold máximo con beneficio | Nº thresholds con beneficio |
|---|---:|---:|---:|
| MLP | 0,01 | 0,36 | 36 |
| Boosting | 0,02 | 0,35 | 34 |

Interpretación:

- ambos modelos tienen utilidad potencial en thresholds bajos-medios,
- fuera de ese rango puede no compensar generar alertas,
- el threshold clínico debe fijarse según capacidad y coste de errores.

## Auditoría interseccional

Se cruzan:

```text
zona x nivel_educativo
```

Resultados:

| Métrica | Valor |
|---|---:|
| Gap máximo de recall | 0,160 |
| Gap máximo de precisión | 0,159 |
| Peor subgrupo por recall | Urbana + Primaria |
| Recall peor subgrupo | 0,566 |
| n peor subgrupo | 1381 |

Subgrupos relevantes:

| Subgrupo | n | Prevalencia | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Urbana + Primaria | 1381 | 0,202 | 0,552 | 0,566 | 0,559 |
| Semiurbana + Primaria | 650 | 0,174 | 0,500 | 0,575 | 0,535 |
| Urbana + Sin estudios | 542 | 0,175 | 0,575 | 0,726 | 0,642 |
| Rural + Secundaria | 793 | 0,197 | 0,613 | 0,641 | 0,627 |

Conclusión:

```text
No hay colapso en ningún subgrupo, pero sí variabilidad que debe vigilarse.
```

## Bootstrap de diferencias

Comparación:

```text
MLP - Boosting
```

Resultado:

| Diferencia | Media | IC 95% inferior | IC 95% superior | Incluye 0 |
|---|---:|---:|---:|---|
| Delta AUC | -0,0037 | -0,0070 | -0,0005 | No |
| Delta F1 | -0,0150 | -0,0258 | -0,0040 | No |

Interpretación:

```text
En esta ejecución, Boosting supera a MLP en AUC y F1 de forma consistente, aunque con diferencia pequeña.
```

## Stress test de prevalencia

La precisión depende de la prevalencia. Por eso se simulan poblaciones con prevalencias distintas.

| Prevalencia objetivo | PPV medio | Recall medio | NPV medio | F1 medio |
|---:|---:|---:|---:|---:|
| 5% | 0,213 | 0,602 | 0,977 | 0,315 |
| 10% | 0,364 | 0,601 | 0,952 | 0,453 |
| 20% | 0,563 | 0,602 | 0,899 | 0,582 |

Lectura:

- el recall se mantiene estable,
- la precisión cae cuando la prevalencia baja,
- el NPV sube en prevalencias bajas,
- el rendimiento dependería del hospital o población de despliegue.

Frase fuerte para presentación:

```text
El modelo no debe transportarse a otra población sin recalibración.
```

## Ablation study

Se retiran bloques de variables para ver qué sostiene el rendimiento.

| Escenario | Nº variables | AUC | Precisión | Recall | F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| Completo sin leakage | 30 | 0,843 | 0,552 | 0,602 | 0,576 | +0,000 |
| Sin socioeconómicas | 25 | 0,831 | 0,539 | 0,580 | 0,559 | -0,017 |
| Sin comorbilidades | 24 | 0,829 | 0,532 | 0,578 | 0,554 | -0,022 |
| Sin genéticas | 23 | 0,774 | 0,487 | 0,463 | 0,475 | -0,101 |
| Solo clínicas | 18 | 0,752 | 0,469 | 0,419 | 0,443 | -0,133 |

Conclusión:

- las socioeconómicas aportan algo, pero no son imprescindibles,
- las comorbilidades aportan señal, pero quitarlas solo produce una caída moderada,
- las genéticas aportan mucho más,
- solo clínicas reduce bastante el rendimiento,
- la disponibilidad real de genética debe discutirse antes de producción.

## Implementación clínica

Flujo:

```text
Paciente -> datos disponibles -> cálculo de riesgo -> alerta -> revisión médica -> decisión final
```

El modelo:

- calcula una probabilidad de riesgo,
- compara contra un threshold,
- genera una alerta,
- prioriza revisión.

El médico:

- interpreta la alerta,
- revisa contexto clínico,
- decide prueba confirmatoria,
- toma la decisión final.

Punto clave:

```text
La decisión final no la toma el modelo.
```

## Uso recomendado y no recomendado

Uso recomendado:

| Uso | Motivo |
|---|---|
| Cribado | Permite detectar más casos potenciales |
| Priorización | Ordena pacientes según riesgo |
| Alerta temprana | Señala perfiles que merecen revisión |
| Apoyo médico | Complementa, no sustituye, el juicio clínico |

Uso no recomendado:

| Uso | Riesgo |
|---|---|
| Diagnóstico automático | El rendimiento no es suficiente |
| Descarte definitivo | Existen falsos negativos |
| Decisiones sin supervisión | Riesgo clínico y ético |
| Producción sin validación externa | Puede fallar en otra población |

## Limitaciones principales

Limitaciones que deben decirse en la presentación:

1. Dataset sintético.
2. Falta validación externa.
3. Falta validación temporal real.
4. Algunas variables pueden no estar disponibles en todos los hospitales.
5. Las variables genéticas podrían no existir antes del diagnóstico.
6. Las variables socioeconómicas requieren revisión ética.
7. La prevalencia afecta PPV, NPV y carga asistencial.
8. El modelo no alcanza fiabilidad de diagnóstico.

Lejos de debilitar el proyecto, estas limitaciones muestran madurez.

## Cómo explicar por qué la diferencia entre modelos es pequeña

La diferencia es pequeña porque probablemente el límite no está en el algoritmo.

Posibles razones:

- las clases se solapan,
- los datos tienen ruido,
- faltan señales clínicas más específicas,
- el dataset es tabular,
- HistGradientBoosting ya es fuerte en tabular,
- la MLP no tiene una ventaja clara en este tipo de datos,
- se eliminaron correctamente variables con fuga.

Frase útil:

```text
No se debe forzar una victoria artificial de la MLP; la decisión debe apoyarse en el comportamiento real de los datos.
```

## Qué habría que hacer para producción real

Antes de usarlo en un hospital real habría que:

1. Validar en una cohorte externa.
2. Validar temporalmente con fechas reales.
3. Revisar disponibilidad de variables.
4. Confirmar que las variables genéticas son prediagnósticas.
5. Auditar sesgo por subgrupos clínicos y sociales.
6. Fijar threshold con clínicos.
7. Definir circuito de alertas.
8. Monitorizar drift.
9. Recalibrar periódicamente.
10. Medir impacto prospectivo.

Frase de producción:

```text
El sistema requeriría monitorización continua y recalibración periódica para mantener su rendimiento en producción.
```

## Estructura recomendada de presentación

Una presentación final sólida podría tener entre 14 y 18 diapositivas.

### Diapositiva 1: título

Título:

```text
Predicción de diagnóstico de cáncer mediante Machine Learning y MLP
```

Subtítulo:

```text
Evaluación técnica, clínica y operativa de un sistema de apoyo al cribado
```

Mensaje:

```text
No es diagnóstico automático; es priorización clínica.
```

### Diapositiva 2: problema

Contenido:

- necesidad de priorizar pacientes,
- riesgo de saturación,
- importancia de detectar casos positivos,
- clase positiva `cancer = 1`.

Mensaje:

```text
El objetivo es detectar señal útil para priorizar revisión.
```

### Diapositiva 3: datos

Contenido:

- 50.001 pacientes,
- 6 fuentes,
- prevalencia 19,29%,
- dataset tabular multimodal.

Tabla sugerida:

| Bloque | Ejemplos |
|---|---|
| Bioquímico | glucosa, colesterol, triglicéridos |
| Clínico | diabetes, hipertensión, obesidad |
| Genético | mutaciones |
| Sociodemográfico | edad, zona, educación |

### Diapositiva 4: fuga de datos

Contenido:

- variables eliminadas,
- comparación con y sin leakage.

Mensaje:

```text
Eliminar leakage baja métricas, pero hace el sistema realista.
```

### Diapositiva 5: pipeline

Contenido:

- split estratificado,
- escalado,
- one-hot encoding,
- modelos,
- ajuste de umbral.

Mensaje:

```text
Test queda reservado para evaluación final.
```

### Diapositiva 6: modelos

Contenido:

- Logistic Regression,
- Random Forest,
- Extra Trees,
- HistGradientBoosting,
- MLP.

Mensaje:

```text
Se comparan modelos clásicos y red neuronal bajo una metodología común.
```

### Diapositiva 7: resultados finales

Tabla:

| Modelo | Threshold | Precisión | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Boosting | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 |
| MLP | 0,32 | 0,522 | 0,630 | 0,571 | 0,840 |

Mensaje:

```text
Boosting gana por poco; MLP tiene más recall.
```

### Diapositiva 8: calibración

Contenido:

- Brier score,
- Boosting calibrado,
- probabilidades más interpretables.

Mensaje:

```text
Para informar riesgo, conviene calibrar.
```

### Diapositiva 9: thresholds clínicos

Tabla:

| Política | Threshold | Recall | FP/1.000 | FN/1.000 |
|---|---:|---:|---:|---:|
| Cribado | 0,13 | 0,857 | 290,3 | 27,6 |
| Equilibrado | 0,34 | 0,605 | 92,8 | 76,1 |
| Alta precisión | 0,52 | 0,363 | 30,0 | 122,8 |

Mensaje:

```text
El threshold es una decisión clínica, no solo técnica.
```

### Diapositiva 10: impacto hospitalario

Contenido:

- 210 alertas por 1.000,
- 117 casos detectados,
- 93 falsos positivos,
- 76 falsos negativos.

Mensaje:

```text
La utilidad depende de la capacidad de revisión posterior.
```

### Diapositiva 11: falsos negativos

Contenido:

- FN tienen señales menos extremas,
- menor carga metabólica,
- menor tabaquismo/obesidad,
- menor carga genética.

Mensaje:

```text
Los errores se concentran en casos menos evidentes.
```

### Diapositiva 12: fairness

Contenido:

- modelo completo vs sin socioeconómicas,
- F1 baja poco,
- gap de recall mejora.

Mensaje:

```text
Hay una tensión entre rendimiento y equidad.
```

### Diapositiva 13: validación avanzada

Contenido:

- DCA,
- bootstrap,
- stress test,
- ablation.

Mensaje:

```text
La decisión no se basa solo en una métrica.
```

### Diapositiva 14: implementación clínica

Flujo:

```text
Paciente -> riesgo -> alerta -> revisión médica -> decisión final
```

Mensaje:

```text
El modelo prioriza; el médico decide.
```

### Diapositiva 15: limitaciones

Contenido:

- dataset sintético,
- sin validación externa,
- variables no siempre disponibles,
- dependencia de prevalencia,
- no diagnóstico automático.

Mensaje:

```text
El proyecto es viable como apoyo, no como producto clínico final.
```

### Diapositiva 16: conclusión final

Conclusión:

```text
Viabilidad moderada.
HistGradientBoosting calibrado recomendado.
Uso como apoyo al cribado/priorización.
Validación externa necesaria antes de producción.
```

## Guion oral recomendado

Inicio:

> El objetivo del proyecto no es diagnosticar cáncer automáticamente, sino estudiar si los datos tabulares disponibles contienen señal suficiente para priorizar pacientes que deberían revisarse antes.

Datos:

> Partimos de seis bloques de información y un dataset final de 50.001 pacientes, con una prevalencia de cáncer del 19,29%. Como el problema está desbalanceado, no usamos accuracy como métrica principal.

Leakage:

> Una parte crítica fue eliminar variables post-diagnóstico. Cuando se incluyen, el AUC llega prácticamente a 1, pero eso sería irreal. Por eso se eliminan y se trabaja con 30 variables válidas.

Modelos:

> Se compararon modelos clásicos y una MLP. La MLP está regularizada y entrenada con validación interna, pero en este problema tabular el Boosting resulta ligeramente más defendible.

Resultados:

> El Boosting obtiene F1 0,576 y AUC 0,843; la MLP obtiene F1 0,571 y AUC 0,840. La diferencia es pequeña, así que la conclusión es prudente.

Clínica:

> El paso importante es traducir métricas a políticas. Para cribado podemos usar un umbral bajo y detectar más casos; para priorización usamos un umbral equilibrado y controlamos la carga médica.

Final:

> La herramienta sería útil como apoyo al cribado o priorización, siempre con revisión médica posterior. No debe usarse para diagnóstico automático y requeriría validación externa antes de producción.

## Preguntas difíciles y respuestas

### ¿Por qué no gana claramente la red neuronal?

Porque el dataset es tabular y los modelos de boosting suelen ser muy fuertes en este tipo de datos. Además, probablemente el límite principal está en la información disponible, no en la arquitectura. La MLP tiene algo más de recall, pero el Boosting ofrece mejor F1, AUC, precisión, simplicidad e interpretabilidad práctica.

### ¿Por qué no usas accuracy como métrica principal?

Porque el problema está desbalanceado. Si solo se mira accuracy, un modelo puede parecer bueno por acertar muchos negativos, aunque falle demasiados positivos. En cribado interesa especialmente recall, F1 y análisis de falsos negativos.

### ¿Por qué calibrar el modelo?

Porque si se comunica riesgo, la probabilidad debe ser interpretable. Un modelo puede ordenar bien pacientes y aun así estar mal calibrado. La calibración mejora el Brier score de Boosting de 0,156 a 0,110.

### ¿Qué threshold usarías?

Depende del uso. Para cribado usaría un threshold bajo, 0,13, porque maximiza detección. Para priorización general usaría 0,34, porque equilibra F1 y carga asistencial. No hay un único threshold universal.

### ¿Qué pasa con los falsos negativos?

Son el error clínicamente más sensible. El análisis muestra que tienen señales menos extremas: menor carga metabólica, menor tabaquismo/obesidad y menor presencia de mutaciones. Eso sugiere que para reducirlos harían falta mejores variables clínicas, no solo otro algoritmo.

### ¿El modelo tiene sesgo?

Hay variabilidad por subgrupos, aunque no se observa colapso extremo. Al quitar variables socioeconómicas baja algo el rendimiento pero mejora el gap de recall. Esto muestra una tensión real entre rendimiento y equidad.

### ¿Se puede usar en un hospital?

No directamente. Harían falta validación externa, validación temporal real, revisión de disponibilidad de variables, integración clínica, monitorización y recalibración periódica.

### ¿Qué aporta el proyecto si el F1 es solo 0,58?

Aporta una evaluación realista. En medicina no siempre se obtiene un clasificador perfecto. El valor está en mostrar que hay señal útil para priorización, pero no suficiente para diagnóstico automático.

### ¿Por qué eliminar variables que daban métricas mejores?

Porque eran variables con riesgo de ser posteriores al diagnóstico. Incluirlas habría generado una evaluación inflada y no aplicable a cribado.

### ¿Cuál es la principal mejora futura?

Mejores variables: síntomas, antecedentes familiares, marcadores tumorales, imagen médica y datos longitudinales. Cambiar de algoritmo probablemente aportaría menos que mejorar la señal clínica.

## Frases clave para cerrar

Frases útiles:

```text
El proyecto prioriza realismo clínico sobre métricas artificialmente altas.
```

```text
La decisión final no es un modelo, sino una política de uso.
```

```text
El threshold debe elegirse según el objetivo clínico y la capacidad asistencial.
```

```text
La calibración convierte el score en una probabilidad más defendible.
```

```text
El sistema no diagnostica: alerta y prioriza.
```

```text
Antes de producción harían falta validación externa, monitorización y recalibración.
```

## Conclusión para tribunal

La conclusión final debe sonar así:

> El proyecto demuestra una viabilidad moderada para usar modelos tabulares en apoyo al cribado o priorización de cáncer. Tras eliminar leakage, ajustar umbrales de forma simétrica, calibrar probabilidades y analizar impacto clínico, el modelo recomendado es HistGradientBoosting calibrado. Su rendimiento no permite diagnóstico automático, pero sí puede generar alertas útiles para revisión médica. El siguiente paso real sería validación externa y prospectiva en una cohorte clínica independiente.

## Checklist antes de presentar

Comprobar:

- el README principal está actualizado,
- este README extendido está disponible,
- los tres notebooks están en orden,
- los outputs importantes están ejecutados o reproducibles,
- las figuras están en `figures/`,
- se entiende la diferencia entre modelo crudo y calibrado,
- se explica leakage con claridad,
- se evita decir "diagnóstico automático",
- se insiste en "apoyo al cribado/priorización",
- se citan limitaciones sin esconderlas.

## Orden recomendado de lectura

Para preparar la defensa:

1. Leer el resumen ejecutivo de `README.md`.
2. Revisar la decisión final y policy card.
3. Leer este README extendido.
4. Preparar diapositivas siguiendo la estructura propuesta.
5. Practicar las preguntas difíciles.

## Cierre final

Con el README principal y este README extendido se puede construir una presentación completa:

- el README principal aporta el informe técnico,
- este README aporta la narrativa de defensa,
- los notebooks aportan evidencia reproducible.

El punto fuerte del proyecto no es conseguir métricas perfectas. El punto fuerte es que conecta modelado, validación, ética, umbrales, carga hospitalaria y limitaciones reales en una historia coherente.
