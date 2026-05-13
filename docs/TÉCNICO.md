# Caso cáncer: predicción de diagnóstico mediante Machine Learning y MLP
## Technical Report & Clinical Evaluation

Este proyecto estudia la viabilidad de anticipar un diagnóstico de cáncer usando datos tabulares multimodales de pacientes. El objetivo no es construir un diagnóstico automático, sino evaluar si los datos disponibles permiten crear una herramienta de apoyo al cribado o a la priorización clínica.

El trabajo compara modelos clásicos de Machine Learning frente a una red neuronal multicapa, cuidando especialmente cuatro aspectos:

- evitar fuga de datos,
- evaluar correctamente la clase positiva `cancer = 1`,
- ajustar umbrales de decisión sin usar test,
- traducir el rendimiento técnico a una lectura clínica y operativa.

El análisis principal está en [`modelo.ipynb`](modelo.ipynb).

La evaluación clínica ampliada está en [`evaluacion_clinica.ipynb`](evaluacion_clinica.ipynb).

La validación clínica avanzada está en [`validacion_clinica.ipynb`](validacion_clinica.ipynb).

Para preparar la exposición final, este informe se complementa con [`DEFENSA.md`](DEFENSA.md), que contiene la narrativa de defensa, estructura sugerida de diapositivas, guion oral y preguntas difíciles.

## Decisión final

El modelo recomendado es:

```text
HistGradientBoosting calibrado
```

La recomendación se basa en una decisión multicriterio:

| Criterio | Mejor opción | Motivo |
|---|---|---|
| F1 en test con umbral optimizado | `HistGradientBoosting` | Mejor equilibrio precisión-recall, aunque por margen pequeño |
| AUC-ROC | `HistGradientBoosting` / calibrado | Capacidad de ordenación ligeramente superior |
| Calibración de probabilidades | `HistGradientBoosting calibrado` | Mejor Brier score tras calibración |
| Simplicidad | `HistGradientBoosting` | Menos complejo que una red neuronal |
| Interpretabilidad práctica | `HistGradientBoosting` + `Extra Trees`/SHAP | Más defendible en datos tabulares |
| Recall alto | Política de umbral sensible | Se consigue bajando el umbral, no cambiando necesariamente de modelo |
| Uso operativo | `HistGradientBoosting calibrado` | Buen rendimiento, probabilidades más fiables y umbral ajustable |

La diferencia de F1 entre `HistGradientBoosting` y la MLP es pequeña. Por tanto, la conclusión no debe presentarse como una superioridad rotunda del modelo clásico, sino como una recomendación práctica: el Boosting ofrece rendimiento igual o ligeramente superior, menor complejidad y, tras calibración, probabilidades más defendibles.

Uso recomendado:

- apoyo al cribado,
- priorización de pacientes,
- selección para revisión clínica adicional,
- generación de alertas de riesgo.

Uso no recomendado:

- diagnóstico automático,
- sustitución de criterio médico,
- decisión clínica sin validación externa.

## Objetivo

Construir un pipeline completo que permita responder:

- si los datos disponibles contienen señal útil para anticipar cáncer,
- qué modelo ofrece mejor equilibrio entre precisión y recall,
- si compensa usar una MLP frente a modelos clásicos,
- qué umbral conviene según la política clínica,
- cómo cambia el rendimiento con leakage,
- si las probabilidades estimadas están bien calibradas,
- qué limitaciones tendría un sistema de cribado basado en estos datos.

## Datos

Los datos proceden de seis tablas del esquema `CASOCANCER`, exportadas posteriormente a CSV locales en la carpeta `data/`.

| Tabla local | Filas | Columnas | Contenido |
|---|---:|---:|---|
| `bioquimicos.csv` | 50.001 | 8 | Variables analíticas |
| `clinicos.csv` | 50.001 | 8 | Diagnósticos y antecedentes clínicos |
| `geneticos.csv` | 50.001 | 8 | Mutaciones genéticas |
| `economicos.csv` | 50.001 | 6 | Variables económicas/asistenciales |
| `generales.csv` | 50.001 | 5 | Hábitos y variables generales |
| `sociodemograficos.csv` | 50.001 | 8 | Edad, educación, ingresos, zona, etc. |

Tras unir por `paciente_id`, el dataset completo queda como:

```text
50.001 filas x 38 columnas
```

La variable objetivo es:

```python
cancer
```

Distribución de clases:

| Dataset | n | Positivos `cancer = 1` | Prevalencia |
|---|---:|---:|---:|
| Total | 50.001 | 9.644 | 19,29% |
| Train | 40.000 | 7.715 | 19,29% |
| Test | 10.001 | 1.929 | 19,29% |

El problema está desbalanceado: aproximadamente uno de cada cinco pacientes pertenece a la clase positiva. Por eso la accuracy no es suficiente para evaluar el sistema.

## Fuga de datos

Durante el análisis se detectaron variables que hacían que los modelos parecieran demasiado buenos:

- `coste_total`,
- `coste_farmaco`,
- `num_ingresos`,
- `dias_hospital`,
- `vive`.

Estas variables tienen alto riesgo de ser posteriores al diagnóstico o consecuencia del tratamiento/evolución. Por tanto, no son válidas para un escenario de predicción previa.

También se excluyó `alcohol`, porque el nuevo metadata indica que es una variable constante y no informativa.

Se excluyeron junto con:

- `paciente_id`, por ser identificador,
- `cancer`, por ser la variable objetivo.

Después de excluirlas:

```text
Features usadas: 30
Features tras preprocesamiento: 45
```

Esta decisión es central. Al eliminar estas variables, las métricas bajan, pero el estudio se vuelve más realista y clínicamente defendible.

### Validación empírica del leakage

En `evaluacion_clinica.ipynb` se comparó el modelo limpio frente a un modelo contaminado con variables post-diagnóstico usando el mismo umbral operativo `0.65`.

| Escenario | Features | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sin leakage | 30 | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 |
| Con variables post-diagnóstico | 36 | 0,65 | 0,990 | 0,970 | 0,980 | 0,999 | 0,993 |

La diferencia confirma que las variables post-diagnóstico inflan artificialmente el rendimiento. El modelo contaminado no es válido para cribado.

## Preprocesamiento

El pipeline de preparación hace:

- separación `X` / `y`,
- exclusión de identificador y variables con fuga,
- split estratificado 80/20,
- escalado de variables numéricas con `StandardScaler`,
- codificación de categóricas con `OneHotEncoder`,
- ajuste del preprocesador solo en train para evitar leakage.

Variables categóricas detectadas:

```text
tipo_seguro
actividad_fisica
nivel_educativo
nivel_ingresos
zona
estado_civil
```

## Modelos clásicos evaluados

Se entrenaron cuatro modelos baseline:

| Modelo | Motivo |
|---|---|
| Logistic Regression | Baseline lineal, rápido e interpretable |
| Random Forest | Ensemble robusto de árboles |
| HistGradientBoosting | Boosting eficiente en datos tabulares |
| Extra Trees | Ensemble aleatorizado útil para comparación e interpretación |

Resultados iniciales en test usando `threshold = 0.50`:

| Modelo | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,441 | 0,754 | 0,557 | 0,843 | 0,768 |
| Logistic Regression | 0,435 | 0,759 | 0,553 | 0,841 | 0,763 |
| Extra Trees | 0,577 | 0,502 | 0,537 | 0,831 | 0,833 |
| Random Forest | 0,676 | 0,361 | 0,471 | 0,835 | 0,843 |

El mejor baseline inicial por F1 fue:

```text
HistGradientBoosting
```

Random Forest tiene mayor accuracy, pero recall bajo. En cribado oncológico eso es problemático porque deja más positivos sin detectar.

## Red neuronal MLP

Se definió una MLP con Keras 3 usando backend Torch, porque el entorno usa Python 3.14 y TensorFlow no dispone de paquete compatible.

Arquitectura:

```text
Input: 45 features

Dense 128
BatchNormalization
ReLU
Dropout 0.25

Dense 64
BatchNormalization
ReLU
Dropout 0.25

Dense 32
BatchNormalization
ReLU
Dropout 0.20

Dense 1
Sigmoid
```

Parámetros:

```text
Total params: 17.281
Trainable params: 16.833
Non-trainable params: 448
```

La arquitectura se mantuvo moderada para evitar sobreajuste en un problema tabular con 46 columnas tras preprocesamiento.

Entrenamiento:

- validación interna estratificada,
- `EarlyStopping(patience=12, restore_best_weights=True)`,
- `ReduceLROnPlateau(patience=6, factor=0.5)`,
- `class_weight` para compensar desbalance.

Pesos de clase:

```text
0: 0,619
1: 2,592
```

Esto penaliza más los errores sobre pacientes con cáncer.

## Ajuste de umbral

No se usa directamente `threshold = 0.50`, porque en un problema desbalanceado puede no ser óptimo.

Se barren umbrales:

```text
0.10 a 0.90, paso 0.01
```

El umbral se elige sobre validación interna, nunca sobre test.

### Corrección metodológica importante

En una comparación inicial, la MLP usaba un umbral optimizado en validación, mientras que `HistGradientBoosting` se evaluaba con `threshold = 0.50`. Esa comparación no era totalmente simétrica.

Se corrigió optimizando también el umbral del mejor baseline clásico con la misma regla:

```text
elegir el threshold que maximiza F1 en validación interna
```

Resultado en validación:

| Modelo | Threshold | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0,65 | 0,560 | 0,591 | 0,575 |
| MLP | 0,32 | 0,524 | 0,620 | 0,568 |

La tabla de valores exactos puede variar ligeramente si se reentrena la MLP, porque la red neuronal tiene cierta variabilidad. El resultado estable es que `HistGradientBoosting` queda como opción operativa recomendada.

## Evaluación final en test

La evaluación final se realiza en test, que no se usa para entrenar ni para elegir umbrales.

Resultados finales con umbrales optimizados:

| Modelo | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 |
| MLP | 0,32 | 0,522 | 0,630 | 0,571 | 0,840 | 0,817 |

Interpretación:

- `HistGradientBoosting` gana por F1, AUC, precisión y accuracy.
- La diferencia de F1 es pequeña.
- La MLP mantiene mayor recall.
- El modelo clásico es preferible por rendimiento similar o ligeramente mejor, simplicidad e interpretabilidad.

La conclusión correcta no es que el Boosting sea muy superior, sino que es la opción más defendible para un sistema tabular de apoyo clínico.

## Calibración de probabilidades

Una buena AUC no garantiza que las probabilidades sean interpretables. Si el modelo predice riesgo 30%, esa cifra debería aproximarse al riesgo observado real.

En la evaluación clínica se detectó que las probabilidades crudas de `HistGradientBoosting` estaban peor calibradas que las de la MLP. Para corregirlo se añadió:

```python
CalibratedClassifierCV(method="sigmoid", cv=5)
```

La calibración se ajusta solo con train. El test sigue reservado para evaluación.

Resultado observado:

| Modelo | Brier score | AUC-ROC |
|---|---:|---:|
| HistGradientBoosting calibrado | 0,110 | 0,844 |
| MLP | 0,154 | 0,839 |
| HistGradientBoosting sin calibrar | 0,156 | 0,843 |

Además, la versión calibrada obtuvo:

| Modelo | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting calibrado | 0,29 | 0,512 | 0,664 | 0,578 | 0,844 | 0,813 |

Interpretación:

- la calibración mejora mucho el Brier score,
- mantiene AUC similar o ligeramente superior,
- no perjudica F1,
- cambia la escala de probabilidades y por eso cambia el threshold óptimo.

Por tanto:

- si se comunica una probabilidad de riesgo, conviene usar `HistGradientBoosting calibrado`,
- si solo se comunica alerta/no alerta, puede usarse el modelo operativo con umbral validado.

## Políticas clínicas de umbral

El modelo no tiene un único uso posible. El umbral debe elegirse según el objetivo clínico.

En `evaluacion_clinica.ipynb` se evaluaron tres políticas sobre el modelo operativo:

| Política | Objetivo | Threshold | TP | FP | FN | TN | Precisión | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alta sensibilidad | Recall >= 0,85 | 0,37 | 1657 | 2911 | 272 | 5161 | 0,363 | 0,859 | 0,510 |
| Equilibrado | Máximo F1 | 0,65 | 1161 | 942 | 768 | 7130 | 0,552 | 0,602 | 0,576 |
| Alta precisión | Precisión >= 0,70 | 0,80 | 656 | 268 | 1273 | 7804 | 0,710 | 0,340 | 0,460 |

Lectura:

- bajar el umbral aumenta detección, pero genera más falsas alarmas,
- subir el umbral aumenta precisión, pero deja más casos sin alerta,
- el umbral equilibrado maximiza F1, pero no tiene por qué ser el óptimo clínico universal.

## Simulación operativa

Con política equilibrada (`threshold = 0.65`), por cada 1.000 pacientes:

| Métrica operativa | Valor aproximado |
|---|---:|
| Alertas totales | 210 |
| Verdaderos positivos detectados | 116 |
| Falsos positivos | 94 |
| Falsos negativos | 77 |

Esto permite estimar carga asistencial. El sistema no diagnostica: genera alertas para revisión posterior.

## Ficha operativa del sistema

Esta sección resume cómo se usaría el sistema en un escenario operativo realista. La ficha se construye con el `HistGradientBoosting calibrado`, porque es la versión más adecuada cuando se quiere comunicar una probabilidad de riesgo y no solo una etiqueta binaria.

La salida del sistema debe interpretarse así:

```text
paciente -> modelo -> alerta de riesgo -> prueba confirmatoria / revisión clínica
```

No se propone como diagnóstico automático. La recomendación es usarlo como herramienta de apoyo para cribado, priorización o revisión clínica.

| Elemento | Decisión final |
|---|---|
| Modelo recomendado | `HistGradientBoosting calibrado` |
| Uso previsto | Alerta de riesgo para priorizar revisión |
| Decisión posterior | Prueba confirmatoria o valoración clínica |
| Tipo de salida | Probabilidad calibrada + alerta según umbral |
| No debe usarse para | Diagnóstico automático, descarte definitivo o sustitución del criterio médico |

Políticas operativas con el modelo calibrado:

| Política | Threshold | Precisión | Recall | F1 | Detectados por 1.000 | Falsos positivos por 1.000 | Falsos negativos por 1.000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cribado | 0,13 | 0,363 | 0,857 | 0,510 | 165,3 | 290,3 | 27,6 |
| Equilibrado | 0,34 | 0,557 | 0,605 | 0,580 | 116,8 | 92,8 | 76,1 |
| Alta precisión | 0,52 | 0,700 | 0,363 | 0,478 | 70,1 | 30,0 | 122,8 |

Lectura clínica:

- si el objetivo principal es no perder casos, se usaría la política de cribado (`threshold = 0.13`),
- si se busca equilibrio entre detección y carga asistencial, se usaría la política equilibrada (`threshold = 0.34`),
- si la capacidad de revisión es limitada y se quieren pocas falsas alarmas, se usaría alta precisión (`threshold = 0.52`).

La política recomendada para presentar como decisión general es la equilibrada, porque mantiene el mejor F1 observado con probabilidades calibradas. Sin embargo, en un contexto clínico real podría preferirse cribado si el coste de un falso negativo se considera mucho mayor que el coste de una falsa alarma.

Riesgos principales:

- falsos positivos: pueden generar ansiedad, pruebas adicionales y carga asistencial,
- falsos negativos: son el riesgo clínico más importante, porque el sistema no alerta algunos casos reales,
- sesgo: el rendimiento puede variar por subgrupos sociales o geográficos,
- dependencia de prevalencia: si cambia la prevalencia real, cambian los valores predictivos,
- dependencia del circuito clínico: una alerta solo es útil si existe una prueba confirmatoria posterior.

## Análisis de errores: falsos negativos

Se añadió un análisis específico de falsos negativos en `evaluacion_clinica.ipynb`. Este análisis compara los pacientes con cáncer detectados por el modelo (`TP`) frente a los pacientes con cáncer que el modelo no detecta (`FN`).

Se priorizan los falsos negativos porque son el error clínicamente más sensible: representan pacientes enfermos que no recibirían alerta del sistema.

Resumen del análisis:

| Grupo | Casos analizados |
|---|---:|
| Verdaderos positivos | 1161 |
| Falsos negativos | 768 |

Variables numéricas con mayores diferencias medias entre FN y TP:

| Variable | Media TP | Media FN | Diferencia FN - TP |
|---|---:|---:|---:|
| `trigliceridos` | 172,1 | 155,1 | -16,9 |
| `glucosa` | 112,2 | 101,9 | -10,3 |
| `colesterol` | 203,7 | 194,2 | -9,4 |
| `plaquetas` | 254,8 | 253,1 | -1,7 |
| `edad` | 56,7 | 55,4 | -1,4 |

También aparecen diferencias relevantes en variables binarias:

| Variable | Media TP | Media FN | Lectura |
|---|---:|---:|---|
| `obesidad` | 0,659 | 0,380 | Los FN tienen menor presencia de obesidad |
| `fumador` | 0,682 | 0,440 | Los FN tienen menor proporción de fumadores |
| `mut_BRCA1` | 0,295 | 0,070 | Los FN tienen mucha menor señal genética BRCA1 |
| `mut_KRAS` | 0,321 | 0,134 | Los FN tienen menor señal genética KRAS |
| `mut_TP53` | 0,309 | 0,145 | Los FN tienen menor señal genética TP53 |

Variables categóricas destacadas:

| Variable | Categoría | Proporción TP | Proporción FN | Diferencia FN - TP |
|---|---|---:|---:|---:|
| `tipo_seguro` | Privado | 0,511 | 0,246 | -0,265 |
| `tipo_seguro` | Público | 0,303 | 0,555 | +0,252 |
| `actividad_fisica` | Baja | 0,633 | 0,499 | -0,134 |
| `actividad_fisica` | Moderada | 0,271 | 0,342 | +0,071 |
| `actividad_fisica` | Alta | 0,096 | 0,159 | +0,063 |

Insights principales:

- los falsos negativos parecen tener señales clínicas menos extremas que los verdaderos positivos,
- el modelo detecta mejor los casos con perfiles más marcados: mayor carga metabólica, más tabaquismo, más obesidad o más mutaciones relevantes,
- una parte de los FN puede corresponder a casos con presentación más silenciosa o menos evidente en las variables disponibles,
- la diferencia por `tipo_seguro` refuerza la necesidad de revisar sesgo y dependencia de variables socioeconómicas.

Conclusión del error analysis:

El problema no parece ser solo de algoritmo. Los FN se concentran en casos con señales predictivas más débiles. Para reducirlos de forma importante harían falta mejores variables clínicas, como síntomas, antecedentes familiares, marcadores tumorales, imagen médica o analíticas longitudinales.

## Mitigación de fairness

Además de auditar el rendimiento por subgrupos, se añadió una mitigación concreta: entrenar una versión del modelo sin variables socioeconómicas.

Variables retiradas:

```text
nivel_ingresos
tipo_seguro
nivel_educativo
zona
estado_civil
```

Comparación:

| Escenario | Features | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy | Gap recall | Gap precisión |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Completo sin leakage | 30 | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 | 0,089 | 0,081 |
| Sin variables socioeconómicas | 25 | 0,65 | 0,539 | 0,580 | 0,559 | 0,831 | 0,823 | 0,061 | 0,084 |

Impacto:

| Cambio | Valor |
|---|---:|
| Delta F1 | -0,017 |
| Delta gap recall | -0,028 |

Lectura:

- retirar variables socioeconómicas reduce ligeramente el rendimiento,
- el F1 baja de 0,576 a 0,559,
- el AUC baja de 0,843 a 0,831,
- el gap de recall mejora de 0,089 a 0,061,
- la precisión entre grupos queda prácticamente igual.

Conclusión de fairness:

Eliminar variables socioeconómicas reduce algo la capacidad predictiva, pero mejora la equidad medida por diferencia de recall entre subgrupos. Esto deja dos opciones defendibles:

- usar el modelo completo si se prioriza rendimiento y se mantiene auditoría continua,
- usar el modelo sin variables socioeconómicas si se prioriza una política más conservadora desde IA responsable.

Para una presentación académica, la opción más sólida es mostrar ambas y recomendar revisión ética antes de elegir una para producción.

## Validación prospectiva simulada

Se añadió una validación prospectiva simulada para aproximar mejor un escenario de producción. Como el dataset no tiene una fecha clínica real, se usa un orden artificial por `paciente_id`:

```text
primer 80% de pacientes -> pasado
último 20% de pacientes -> futuro
```

Dentro del bloque de pasado, se reserva una parte para optimizar el threshold. El bloque futuro queda como evaluación final simulada.

Resultado:

| Escenario | Train pasado | Test futuro | Prevalencia pasado | Prevalencia futura | Threshold | Precisión | Recall | F1 | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prospective split simulado | 40000 | 10001 | 0,194 | 0,188 | 0,61 | 0,509 | 0,642 | 0,568 | 0,841 | 0,816 |

Lectura:

- el AUC se mantiene en 0,841, muy cerca del test principal,
- el F1 se mantiene en 0,568, también cercano al modelo operativo,
- el recall sube a 0,642, con precisión algo menor,
- no aparece una caída fuerte al evaluar en un bloque separado como futuro.

Conclusión:

El modelo mantiene rendimiento en un escenario más realista que un split aleatorio puro. Aun así, esta validación es simulada: para uso clínico real haría falta una cohorte temporal externa con fechas reales de atención, diagnóstico y seguimiento.

## Análisis coste-beneficio

Se analizó un escenario donde:

```text
coste_FN = 10
coste_FP = 1
```

Es decir, un falso negativo se considera 10 veces más costoso que un falso positivo.

Resultado:

| Threshold | TP | FP | FN | TN | Coste total | Coste por 1.000 |
|---:|---:|---:|---:|---:|---:|---:|
| 0,27 | 1769 | 3815 | 160 | 4257 | 5415 | 541,4 |

Interpretación:

- si se penaliza mucho el falso negativo, el threshold óptimo baja,
- se detectan más casos,
- aumentan las falsas alarmas,
- el umbral de máximo F1 no siempre coincide con el umbral de mínimo coste clínico.

## Validación cruzada

La validación cruzada estratificada se realiza sobre train, manteniendo test reservado para evaluación final.

Se reportan métricas con dos políticas:

| Política | Threshold | AUC media | AUC std | F1 medio | F1 std | Recall medio | Recall std | Precisión media | Precisión std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Threshold 0,50 | 0,50 | 0,839 | 0,005 | 0,548 | 0,008 | 0,748 | 0,013 | 0,432 | 0,010 |
| Threshold operativo | 0,65 | 0,839 | 0,005 | 0,561 | 0,007 | 0,585 | 0,016 | 0,540 | 0,013 |

Interpretación:

- AUC es estable, lo que indica capacidad de ordenación razonable,
- F1 mejora al usar el umbral operativo,
- el recall baja y la precisión sube con el umbral operativo,
- no conviene comparar la métrica final con una CV calculada solo con `threshold = 0.50`.

## Interpretabilidad

`HistGradientBoosting` no expone `feature_importances_` de forma nativa en scikit-learn. Por eso se usan modelos y técnicas auxiliares:

- `Extra Trees` para importancia global aproximada,
- SHAP en modo rápido para explicación global y local.

Variables importantes detectadas:

| Variable | Lectura |
|---|---|
| `fumador` | Factor clínicamente plausible |
| `obesidad` | Factor clínicamente plausible |
| `mut_TP53` | Señal genética relevante si está disponible antes del diagnóstico |
| `mut_BRCA1` | Señal genética relevante si está disponible antes del diagnóstico |
| `mut_KRAS` | Señal genética relevante si está disponible antes del diagnóstico |
| `tipo_seguro` | Variable sensible; puede reflejar acceso o sesgo asistencial |
| `actividad_fisica` | Hábito potencialmente relacionado con riesgo |
| `hipertension` | Comorbilidad |
| `edad` | Factor de riesgo habitual |

Es importante no interpretar estas asociaciones como causalidad. La importancia indica contribución predictiva dentro del dataset.

## Auditoría por subgrupos

Se revisó el rendimiento por grupos socioeconómicos y geográficos disponibles.

Resultados resumidos:

| Subgrupo | n | Prevalencia | Precisión | Recall |
|---|---:|---:|---:|---:|
| Primaria | 2523 | 0,199 | 0,540 | 0,574 |
| Universitario | 2488 | 0,196 | 0,565 | 0,595 |
| Secundaria | 3973 | 0,191 | 0,543 | 0,610 |
| Sin estudios | 1017 | 0,178 | 0,585 | 0,663 |
| Semiurbana | 2569 | 0,180 | 0,512 | 0,598 |
| Urbana | 5432 | 0,195 | 0,556 | 0,599 |
| Rural | 2000 | 0,204 | 0,593 | 0,614 |

Lectura:

- hay variabilidad entre subgrupos,
- no se observa un colapso extremo de rendimiento,
- las diferencias deben vigilarse antes de cualquier uso real,
- variables socioeconómicas requieren revisión ética.

## Validación clínica avanzada

`validacion_clinica.ipynb` amplía la evaluación final con pruebas más cercanas a una decisión clínica real:

- Decision Curve Analysis,
- auditoría interseccional,
- bootstrap de diferencias entre modelos,
- stress test de prevalencia,
- ablation study por bloques de variables.

En una ejecución de verificación:

| Modelo | Threshold | Precisión | Recall | F1 | AUC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Boosting | 0,65 | 0,552 | 0,602 | 0,576 | 0,843 | 0,829 |
| MLP | 0,59 | 0,487 | 0,660 | 0,561 | 0,839 | 0,800 |

### Decision Curve Analysis

La Decision Curve Analysis compara el beneficio neto del modelo frente a dos estrategias simples:

- tratar/revisar a todos,
- no tratar/revisar a nadie.

Resultado:

| Modelo | Threshold mínimo con beneficio | Threshold máximo con beneficio | Nº thresholds con beneficio |
|---|---:|---:|---:|
| MLP | 0,01 | 0,36 | 36 |
| Boosting | 0,02 | 0,35 | 34 |

Interpretación:

- ambos modelos tienen utilidad clínica potencial en rangos bajos-medios de threshold,
- el Boosting mantiene beneficio neto en un rango amplio,
- fuera de esos rangos, el umbral puede generar más daño operativo que beneficio.

### Auditoría interseccional

Se cruzaron `zona` y `nivel_educativo` para comprobar si había subgrupos con peor sensibilidad.

Resumen:

| Métrica | Valor |
|---|---:|
| Gap máximo de recall | 0,160 |
| Gap máximo de precisión | 0,159 |
| Peor subgrupo por recall | Urbana + Primaria |
| Recall peor subgrupo | 0,566 |
| n peor subgrupo | 1381 |

Subgrupos extremos:

| Subgrupo | n | Prevalencia | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Urbana + Primaria | 1381 | 0,202 | 0,552 | 0,566 | 0,559 |
| Semiurbana + Primaria | 650 | 0,174 | 0,500 | 0,575 | 0,535 |
| Urbana + Sin estudios | 542 | 0,175 | 0,575 | 0,726 | 0,642 |
| Rural + Secundaria | 793 | 0,197 | 0,613 | 0,641 | 0,627 |

Interpretación:

- no hay colapso completo en ningún subgrupo,
- sí existe variabilidad relevante,
- el peor recall aparece en `Urbana + Primaria`,
- esto justifica mantener auditoría de sesgo antes de cualquier uso real.

### Bootstrap de diferencias entre modelos

El bootstrap de diferencias `MLP - Boosting` mostró:

| Diferencia | Media | IC 95% inferior | IC 95% superior | Incluye 0 |
|---|---:|---:|---:|---|
| Delta AUC | -0,0037 | -0,0070 | -0,0005 | No |
| Delta F1 | -0,0150 | -0,0258 | -0,0040 | No |

Interpretación:

- los intervalos no incluyen 0,
- en esta ejecución el Boosting supera a la MLP en AUC y F1,
- la diferencia es consistente, pero pequeña.

### Stress test de prevalencia

Se simuló cómo cambian las métricas si la prevalencia poblacional baja o sube. Esto es importante porque precisión y valor predictivo dependen mucho de la prevalencia.

| Prevalencia objetivo | PPV medio | Recall medio | NPV medio | F1 medio | n medio |
|---:|---:|---:|---:|---:|---:|
| 5% | 0,213 | 0,602 | 0,977 | 0,315 | 8496 |
| 10% | 0,364 | 0,601 | 0,952 | 0,453 | 8968 |
| 20% | 0,563 | 0,602 | 0,899 | 0,582 | 9645 |

Interpretación:

- el recall se mantiene alrededor de 0,60,
- la precisión cae mucho cuando la prevalencia baja,
- el NPV es alto en prevalencias bajas,
- el sistema debe recalibrarse o reevaluarse si cambia la población objetivo.

### Ablation study

Se entrenaron variantes retirando bloques de variables para estimar qué señales sostienen el rendimiento.

| Escenario | Nº variables | AUC | Precisión | Recall | F1 | Delta F1 | Delta AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Completo sin leakage | 30 | 0,843 | 0,552 | 0,602 | 0,576 | +0,000 | +0,000 |
| Sin socioeconómicas | 25 | 0,831 | 0,539 | 0,580 | 0,559 | -0,017 | -0,012 |
| Sin comorbilidades | 24 | 0,829 | 0,532 | 0,578 | 0,554 | -0,022 | -0,014 |
| Sin genéticas | 23 | 0,774 | 0,487 | 0,463 | 0,475 | -0,101 | -0,069 |
| Solo clínicas | 18 | 0,752 | 0,469 | 0,419 | 0,443 | -0,133 | -0,091 |

Interpretación:

- quitar variables socioeconómicas tiene un coste moderado,
- quitar comorbilidades también reduce el rendimiento de forma moderada, lo que sugiere posible señal clínica sin dependencia excesiva,
- quitar variables genéticas reduce mucho más el rendimiento,
- usar solo variables clínicas deja un modelo bastante más débil,
- las señales genéticas son importantes en este dataset, pero su disponibilidad prediagnóstica debe revisarse en un escenario real.

Conclusión de la validación avanzada:

El Boosting operativo es la opción más defendible, pero no por una superioridad enorme. Lo importante es que mantiene utilidad neta, rendimiento estable, mejor F1 que la MLP en bootstrap y una degradación comprensible al retirar bloques informativos.

## Implementación en entorno clínico

La validación final se traduce a un flujo operativo realista. El objetivo no es diagnosticar automáticamente, sino priorizar pacientes para revisión médica.

### Flujo operativo

```text
Paciente -> datos disponibles -> cálculo de riesgo -> alerta si riesgo supera el umbral -> revisión médica -> decisión final
```

El modelo actúa como sistema de apoyo. Una alerta indica prioridad de revisión, no confirmación diagnóstica.

### Política de decisión

Se plantean dos escenarios principales:

| Escenario | Threshold | Objetivo | Uso esperado |
|---|---:|---|---|
| Cribado | 0,13 | Maximizar detección | Revisar más pacientes para reducir falsos negativos |
| Priorización | 0,34 | Equilibrar detección y carga | Generar alertas más manejables para revisión clínica |

En cribado se usaría un umbral más bajo para detectar más casos. En priorización clínica se usaría un umbral equilibrado para controlar la carga asistencial.

### Impacto hospitalario

Con la política equilibrada del modelo calibrado, por cada 1.000 pacientes se esperan aproximadamente:

| Resultado operativo | Pacientes por 1.000 |
|---|---:|
| Alertas generadas | 210 |
| Casos detectados | 117 |
| Falsos positivos revisados | 93 |
| Casos no alertados | 76 |

Esto equivale a unos 210 pacientes derivados a revisión por cada 1.000 evaluados. La carga puede ser asumible si la alerta se integra como priorización clínica y no como diagnóstico final.

### Limitaciones y uso correcto

Limitaciones para uso clínico:

- el dataset es sintético,
- falta validación externa en otro hospital o cohorte temporal real,
- algunas variables pueden no estar disponibles en todos los centros,
- el rendimiento depende de la prevalencia de la población evaluada,
- las variables socioeconómicas y genéticas requieren revisión ética y clínica.

El stress test de prevalencia muestra que el comportamiento cambia según el contexto poblacional: con prevalencia del 5%, el PPV medio baja a 0,213; con prevalencia del 20%, sube a 0,563. Esto sugiere que el modelo necesitaría recalibración si se despliega en poblaciones distintas.

Uso recomendado:

- cribado,
- priorización clínica,
- alerta temprana,
- apoyo a revisión médica.

Uso no recomendado:

- diagnóstico automático,
- descarte definitivo de pacientes,
- decisiones clínicas sin supervisión,
- uso en producción sin validación externa.

El sistema requeriría monitorización continua y recalibración periódica para mantener su rendimiento en producción.

## Por qué la mejora entre modelos es pequeña

La diferencia entre `HistGradientBoosting` y MLP es pequeña porque probablemente el límite principal está en la información disponible en las variables, no en el algoritmo.

Factores que explican el techo de rendimiento:

- clases con solapamiento,
- ruido en el dataset,
- ausencia de variables clínicas más específicas,
- eliminación correcta de variables con fuga,
- número de features moderado,
- datos tabulares donde el boosting ya es muy competitivo.

No se debe forzar que la MLP gane. La decisión final se apoya en rendimiento, calibración, simplicidad e interpretabilidad.

## Limitaciones

- Dataset sintético.
- Posible diferencia con pacientes reales.
- Ruido y solapamiento entre clases.
- Falta validación externa.
- Variables genéticas pueden no estar disponibles antes del diagnóstico en todos los contextos.
- Variables socioeconómicas requieren revisión ética.
- El umbral óptimo depende de prevalencia, costes clínicos y capacidad asistencial.
- La calibración mejora probabilidades, pero también debería validarse externamente.
- La herramienta no debe usarse como diagnóstico automático.

## Datos que mejorarían el modelo

Sería útil incorporar:

- antecedentes familiares,
- marcadores tumorales específicos,
- evolución temporal de analíticas,
- síntomas registrados,
- resultados de imagen médica,
- tratamientos previos,
- información longitudinal,
- validación con cohortes reales.

Si las clases se solapan con las variables actuales, cambiar de algoritmo puede aportar poco. La mejora más importante vendría de mejores señales clínicas.

## Reproducibilidad

Entorno usado:

```text
Python 3.14.2
Keras 3
Backend Torch
scikit-learn 1.8.0
pandas
SQLAlchemy
pyodbc
```

Orden recomendado de ejecución:

1. `modelo.ipynb`
2. `evaluacion_clinica.ipynb`
3. `validacion_clinica.ipynb`

Dentro de `modelo.ipynb`:

1. Fase 0: carga/exportación de datos, si no existen los CSV locales.
2. Fase 1: preparación para modelado.
3. Fase 2: modelos ML baseline.
4. Feature importance.
5. Fase 3: definición de la MLP.
6. Fase 4: entrenamiento de la MLP.
7. Fase 5: ajuste de umbrales.
8. Fase 6: evaluación final en test.
9. Fase 7: comparación global y conclusión.

Nota: si se reejecuta la Fase 4, conviene ejecutar antes la Fase 3 para reinicializar la red y evitar entrenar sobre pesos antiguos.

## Conclusión

La viabilidad del sistema es **moderada**.

Los datos permiten cierta anticipación del diagnóstico, con AUC alrededor de 0,84 y F1 alrededor de 0,57. Esto indica capacidad de priorización, pero no suficiente fiabilidad para diagnóstico automático.

La recomendación final es:

```text
HistGradientBoosting calibrado
```

Motivo:

- mantiene el mejor rendimiento operativo,
- mejora la calibración de probabilidades,
- es más simple que la MLP,
- es más adecuado para datos tabulares,
- permite ajustar el umbral según objetivo clínico,
- conserva una narrativa interpretativa más defendible.

El sistema debería plantearse como una herramienta de apoyo al cribado o priorización, siempre con revisión clínica posterior y validación externa antes de cualquier uso real.
