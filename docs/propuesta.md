# Propuesta empresarial, validación clínica y despliegue responsable

## Objetivo del documento

Este documento complementa el análisis técnico del proyecto de predicción de cáncer. Su finalidad es traducir el modelo a una propuesta de valor clínica y empresarial: quién podría usarlo, qué problema operativo resuelve, cómo se mediría su impacto, qué coste tendría su uso y qué pasos serían necesarios antes de plantear una implantación real.

La idea central no es vender un diagnóstico automático, sino una herramienta de apoyo para priorizar revisión clínica.

```text
Modelo de riesgo -> alerta priorizada -> revisión profesional -> decisión clínica documentada
```

El valor del sistema no está en sustituir al médico, sino en ayudar a ordenar pacientes cuando la capacidad asistencial es limitada.

---

## 1. Propuesta de valor empresarial

### Problema que resuelve

Los sistemas sanitarios trabajan con una tensión constante: muchos pacientes, recursos limitados y necesidad de detectar antes los casos que requieren revisión. En oncología, el retraso diagnóstico puede tener consecuencias clínicas, económicas y organizativas relevantes.

El problema práctico no es solo predecir si un paciente tiene cáncer. El problema operativo es:

```text
De todos los pacientes disponibles, identificar cuáles deberían revisarse antes.
```

El modelo propuesto funciona como una capa de priorización. A partir de datos tabulares del paciente, estima una probabilidad de riesgo y genera una alerta cuando supera un umbral definido por la política clínica.

### Quién podría comprar o adoptar el sistema

| Comprador o usuario | Interés principal | Uso esperado |
|---|---|---|
| Hospital público o privado | Reducir demoras y ordenar listas de revisión | Priorización de pacientes en consultas, pruebas o derivaciones |
| Red de atención primaria | Detectar pacientes que requieren evaluación preferente | Alerta temprana antes de derivación especializada |
| Unidad oncológica | Identificar perfiles de alto riesgo ya presentes en el sistema | Apoyo a comités, cribado oportunista o seguimiento |
| Aseguradora sanitaria | Optimizar circuitos y reducir costes por diagnóstico tardío | Segmentación de riesgo y priorización de pruebas confirmatorias |
| Servicio de gestión clínica | Mejorar eficiencia asistencial | Dashboard de carga, alertas y rendimiento del circuito |

La propuesta es especialmente interesante para organizaciones que ya tienen datos estructurados de pacientes y sufren cuellos de botella en revisión, derivación o pruebas diagnósticas.

### Dolor económico y operativo que aborda

El sistema puede ayudar a reducir o controlar varios problemas:

| Dolor actual | Impacto | Cómo ayuda el modelo |
|---|---|---|
| Listas de espera no priorizadas | Pacientes de mayor riesgo pueden revisarse tarde | Ordena pacientes por probabilidad de riesgo |
| Uso ineficiente de especialistas | Tiempo clínico dedicado a casos de menor prioridad | Concentra revisión en alertas de mayor riesgo |
| Retrasos diagnósticos | Mayor coste clínico, terapéutico y reputacional | Facilita detección y revisión temprana |
| Exceso de pruebas sin criterio de riesgo | Coste y saturación | Permite ajustar umbral según capacidad |
| Falta de trazabilidad en priorización | Dificulta auditoría y mejora continua | Documenta alerta, umbral, responsable y resultado |

### KPI de negocio y gestión clínica

Para defender el sistema ante una dirección hospitalaria no basta con AUC o F1. Es necesario medir indicadores de impacto operativo.

| KPI | Qué mide | Por qué importa |
|---|---|---|
| Tiempo medio hasta revisión | Días desde alerta hasta evaluación clínica | Mide si el sistema acelera la atención |
| Alertas por 1.000 pacientes | Carga generada por el modelo | Permite dimensionar recursos |
| Alertas por profesional/día | Carga real de trabajo | Evita saturar equipos |
| Casos positivos detectados por 1.000 pacientes | Rendimiento clínico del circuito | Traduce recall a actividad asistencial |
| Falsos positivos por 1.000 pacientes | Revisiones adicionales no confirmadas | Mide coste operativo |
| Falsos negativos por 1.000 pacientes | Casos que el sistema no alerta | Principal riesgo clínico |
| Coste por caso detectado | Coste total del circuito dividido por casos detectados | Métrica clara para dirección |
| Porcentaje de alertas revisadas en plazo | Cumplimiento operativo | Evalúa adopción real |
| Tasa de aceptación médica de alertas | Porcentaje de alertas consideradas útiles | Mide confianza clínica |

### Carga operativa con la política equilibrada

El proyecto estima que, con la política equilibrada del modelo calibrado, se generarían aproximadamente:

| Métrica por 1.000 pacientes | Valor aproximado |
|---|---:|
| Alertas totales | 210 |
| Casos positivos detectados | 117 |
| Falsos positivos | 93 |
| Falsos negativos | 76 |
| Precisión | 0,557 |
| Recall | 0,605 |
| F1 | 0,580 |

Esto significa que por cada 1.000 pacientes el sistema no pide revisar a todo el mundo. Prioriza aproximadamente a uno de cada cinco pacientes.

Desde una perspectiva empresarial, la pregunta clave es:

```text
¿Tiene el sistema capacidad operativa para absorber 210 alertas por cada 1.000 pacientes?
```

Si cada alerta requiere una revisión inicial de 10 minutos por enfermería o gestor clínico, entonces:

```text
210 alertas x 10 minutos = 2.100 minutos = 35 horas de revisión
```

Por tanto, cada 1.000 pacientes implicarían alrededor de una semana laboral de revisión inicial. Para 10.000 pacientes, serían unas 350 horas de revisión inicial.

Este cálculo es orientativo, pero convierte el modelo en una conversación de capacidad real.

---

## 2. Business case cuantificado

### Supuestos económicos iniciales

El análisis técnico usa una relación de coste:

```text
coste_FN = 10
coste_FP = 1
```

Esta relación expresa que un falso negativo pesa diez veces más que un falso positivo. Para una dirección hospitalaria conviene traducir esta lógica a euros, tiempo y capacidad.

Los siguientes valores son supuestos de trabajo. No representan tarifas oficiales, sino una base para construir escenarios.

| Concepto | Supuesto base |
|---|---:|
| Revisión inicial de alerta por enfermería/gestor | 10 minutos |
| Coste hora enfermería/gestor clínico | 35 € |
| Coste revisión inicial por alerta | 5,83 € |
| Revisión médica de alerta prioritaria | 15 minutos |
| Coste hora médica especialista | 90 € |
| Coste revisión médica por alerta escalada | 22,50 € |
| Prueba confirmatoria media | 250 € |
| Coste incremental de diagnóstico tardío evitable | 5.000 € |

Estos supuestos deberían sustituirse por costes reales del hospital o aseguradora en una fase de implantación.

### Escenario operativo por 10.000 pacientes

Usando la política equilibrada:

| Métrica | Por 1.000 pacientes | Por 10.000 pacientes |
|---|---:|---:|
| Alertas totales | 210 | 2.100 |
| Casos detectados | 117 | 1.170 |
| Falsos positivos | 93 | 930 |
| Falsos negativos | 76 | 760 |

Si cada alerta requiere revisión inicial de 10 minutos:

```text
2.100 alertas x 10 minutos = 21.000 minutos = 350 horas
```

Equivalencia aproximada:

```text
350 horas / 35 horas semanales = 10 semanas-persona
```

Si el circuito se reparte entre 2 profesionales, el volumen de revisión inicial para 10.000 pacientes equivaldría aproximadamente a 5 semanas de trabajo de cada profesional.

### Coste de revisión inicial

```text
2.100 alertas x 5,83 € = 12.243 €
```

Este sería el coste aproximado de la primera revisión administrativa o clínica ligera.

### Coste si todas las alertas pasan a revisión médica

```text
2.100 alertas x 22,50 € = 47.250 €
```

En la práctica no todas las alertas deberían llegar directamente al especialista. Un diseño más eficiente sería:

```text
alerta IA -> revisión inicial -> escalado médico solo si procede
```

Si solo el 50 % de alertas se escala a médico:

```text
1.050 revisiones médicas x 22,50 € = 23.625 €
```

### Coste de pruebas confirmatorias

Si todos los casos escalados reciben prueba confirmatoria:

```text
1.050 pruebas x 250 € = 262.500 €
```

Esto muestra por qué el sistema no debe conectarse automáticamente a pruebas. La alerta debe pasar por revisión profesional. El valor empresarial depende de usar el modelo como priorizador, no como disparador automático de consumo sanitario.

### Beneficio potencial por detección temprana

Si de los 1.170 casos detectados por cada 10.000 pacientes solo un 10 % representa detección clínicamente adelantada respecto al circuito habitual:

```text
117 casos adelantados
```

Si cada diagnóstico adelantado evita un coste incremental medio de 5.000 €:

```text
117 x 5.000 € = 585.000 €
```

Comparado con los costes operativos aproximados:

| Concepto | Coste estimado |
|---|---:|
| Revisión inicial de 2.100 alertas | 12.243 € |
| Revisión médica del 50 % de alertas | 23.625 € |
| Pruebas confirmatorias en alertas escaladas | 262.500 € |
| Coste operativo total estimado | 298.368 € |
| Beneficio potencial por detección adelantada | 585.000 € |
| Balance potencial | +286.632 € |

Con estos supuestos, el ROI aproximado sería:

```text
ROI = (beneficio - coste) / coste
ROI = (585.000 - 298.368) / 298.368 = 0,96
```

Es decir, un retorno potencial del 96 %. Esta cifra no debe presentarse como resultado demostrado, sino como escenario de negocio a validar.

### Lectura empresarial

El sistema es atractivo si se cumplen tres condiciones:

1. La revisión de alertas se integra en un circuito eficiente.
2. No todas las alertas generan pruebas automáticas.
3. Una proporción suficiente de casos detectados representa detección adelantada real.

Si estas condiciones no se cumplen, el modelo puede aumentar carga y costes sin mejorar resultados. Por eso el piloto clínico debe medir impacto real, no solo métricas de clasificación.

---

## 3. Plan de validación clínica real

El proyecto actual usa un dataset sintético. Esto es aceptable para demostrar metodología, pero no permite afirmar validez clínica real. La forma profesional de presentarlo es convertir esa limitación en un roadmap de validación.

### Fase 1. Validación retrospectiva en cohorte real

Objetivo:

```text
Comprobar si el modelo mantiene rendimiento en pacientes reales ya atendidos.
```

Acciones:

- recopilar datos históricos de un hospital o red sanitaria,
- verificar disponibilidad real de cada variable antes del diagnóstico,
- excluir variables posteriores al diagnóstico,
- medir AUC, recall, precisión, F1, calibración y carga de alertas,
- comparar rendimiento por edad, sexo si existe, zona, nivel socioeconómico y disponibilidad genética.

Criterio de avance:

```text
El modelo mantiene rendimiento útil y calibración aceptable sin leakage temporal.
```

### Fase 2. Validación temporal con fechas reales

Objetivo:

```text
Evaluar si el modelo funciona cuando se entrena en el pasado y se prueba en pacientes futuros.
```

Acciones:

- ordenar pacientes por fecha de atención,
- entrenar con periodos anteriores,
- validar en periodos posteriores,
- medir drift de prevalencia, variables y rendimiento,
- recalibrar si cambia la población.

Criterio de avance:

```text
El rendimiento no cae de forma relevante en periodos posteriores.
```

### Fase 3. Validación externa en otro centro

Objetivo:

```text
Comprobar si el sistema generaliza fuera del hospital de origen.
```

Acciones:

- probar el modelo en otro hospital o área sanitaria,
- revisar diferencias de codificación y disponibilidad de variables,
- repetir métricas globales y por subgrupos,
- comparar calibración y carga de alertas.

Criterio de avance:

```text
El modelo mantiene utilidad clínica o puede recalibrarse sin rediseño completo.
```

### Fase 4. Estudio prospectivo silencioso

Objetivo:

```text
Observar el comportamiento del modelo en tiempo real sin intervenir decisiones clínicas.
```

Acciones:

- ejecutar el modelo en pacientes nuevos,
- ocultar la alerta al equipo clínico o marcarla como no vinculante,
- registrar qué habría recomendado,
- comparar con el circuito asistencial real,
- medir cuántos casos habría priorizado antes.

Criterio de avance:

```text
El sistema identifica oportunidades reales de priorización sin generar carga excesiva.
```

### Fase 5. Piloto clínico supervisado

Objetivo:

```text
Usar alertas en un circuito limitado con revisión profesional obligatoria.
```

Acciones:

- seleccionar una unidad, centro o población limitada,
- definir umbral inicial,
- asignar responsables,
- medir tiempos de revisión,
- documentar aceptación o rechazo de alertas,
- auditar falsos positivos y falsos negativos,
- evaluar satisfacción clínica.

Criterio de avance:

```text
El sistema mejora priorización sin deteriorar seguridad, equidad ni carga asistencial.
```

### Fase 6. Despliegue limitado con monitorización

Objetivo:

```text
Implantar el sistema de forma controlada y auditable.
```

Acciones:

- monitorizar rendimiento mensual,
- recalibrar probabilidades si cambia la prevalencia,
- revisar drift de variables,
- mantener auditoría de sesgo,
- registrar decisiones clínicas posteriores,
- activar protocolo de pausa si el rendimiento cae.

Criterio de continuidad:

```text
El modelo mantiene rendimiento, seguridad, trazabilidad y aceptación clínica.
```

---

## 4. Marco regulatorio y ético

### Principios de cumplimiento

Un sistema de apoyo a priorización clínica debe diseñarse bajo principios de prudencia:

- finalidad explícita,
- minimización de datos,
- trazabilidad,
- supervisión humana,
- auditoría periódica,
- explicabilidad suficiente,
- revisión ética de variables sensibles.

### RGPD y protección de datos

Para una implantación real en España o la Unión Europea, el sistema debería considerar:

| Aspecto | Requisito esperado |
|---|---|
| Base legal | Definir fundamento para tratamiento de datos sanitarios |
| Finalidad | Usar datos solo para priorización clínica definida |
| Minimización | Incluir solo variables necesarias para el modelo |
| Seudonimización | Separar identificadores directos de datos analíticos |
| Control de acceso | Limitar acceso a profesionales autorizados |
| Retención | Definir cuánto tiempo se conservan alertas y logs |
| Derechos del paciente | Informar según el marco aplicable y permitir ejercicio de derechos |
| Seguridad | Cifrado, control de accesos y registro de actividad |

### Uso de datos genéticos

Las variables genéticas son predictivamente relevantes en el proyecto, pero su uso exige especial cuidado.

Antes de producción habría que confirmar:

- si las mutaciones están disponibles antes del diagnóstico,
- si se obtuvieron con consentimiento válido,
- si pueden usarse para esta finalidad concreta,
- quién puede acceder a esta información,
- cómo se evita discriminación o uso indebido,
- si existe una versión alternativa del modelo sin genética.

Una opción prudente sería mantener dos versiones:

| Versión | Uso |
|---|---|
| Modelo completo | Contextos donde genética está disponible y autorizada |
| Modelo sin genética | Atención primaria o escenarios con menor disponibilidad/sensibilidad |

### Explicabilidad para clínicos

El sistema no debe limitarse a mostrar una probabilidad. Cada alerta debería acompañarse de una explicación breve y comprensible:

```text
Riesgo estimado: 34 %
Umbral aplicado: 34 %
Circuito: priorización
Factores que más contribuyen: mutación KRAS, tabaquismo, glucosa elevada, edad
```

La explicación no tiene que ser perfecta desde el punto de vista matemático, pero sí suficiente para que el profesional pueda valorar si la alerta tiene sentido.

### Trazabilidad de cada alerta

Cada alerta debería registrar:

| Campo | Descripción |
|---|---|
| ID de alerta | Identificador interno |
| Fecha y hora | Momento de generación |
| Versión del modelo | Modelo exacto utilizado |
| Umbral aplicado | Política clínica usada |
| Probabilidad estimada | Riesgo calculado |
| Variables principales | Factores explicativos relevantes |
| Profesional responsable | Persona o equipo que revisa |
| Decisión posterior | Aceptada, descartada, prueba solicitada, seguimiento |
| Resultado clínico | Confirmación, descarte, pendiente |

Esta trazabilidad permite auditoría, mejora continua y defensa clínica si existe discrepancia.

### Qué hacer si el modelo falla

El sistema debe tener un protocolo de seguridad:

| Situación | Respuesta |
|---|---|
| Caída técnica del modelo | Continuar circuito clínico habitual sin IA |
| Aumento anómalo de alertas | Revisar drift, prevalencia y calibración |
| Bajada de recall | Pausar uso o subir sensibilidad |
| Sesgo detectado en subgrupo | Revisar variables, umbrales y circuito |
| Error clínico relevante | Análisis de incidente y revisión del modelo |
| Datos incompletos | No generar alerta automática o marcar baja confianza |

El sistema debe diseñarse para fallar de forma segura. Si no hay confianza en datos o modelo, debe prevalecer el circuito clínico estándar.

### Clasificación funcional del sistema

La clasificación más prudente del sistema sería:

```text
Software de apoyo a la decisión clínica para priorización, no diagnóstico autónomo.
```

Esto implica:

- el médico conserva la decisión final,
- la alerta no confirma ni descarta cáncer,
- no debe automatizar pruebas o tratamientos sin revisión,
- requiere validación, documentación, monitorización y gestión de riesgos.

---

## 5. Flujo clínico con responsables

### Flujo operativo propuesto

```text
1. Paciente entra en el sistema
2. Se recopilan variables disponibles
3. El modelo calcula probabilidad de riesgo
4. Si supera umbral, se genera alerta
5. Enfermería o gestor clínico revisa la alerta
6. Médico valida o descarta la prioridad
7. Se solicita prueba, consulta o seguimiento si procede
8. Se documenta el resultado
9. El caso se usa para auditoría y mejora continua
```

### Responsables por fase

| Fase | Responsable | Decisión |
|---|---|---|
| Carga de datos | Sistema clínico / IT | Verificar disponibilidad y calidad |
| Cálculo de riesgo | Modelo IA | Generar probabilidad, no diagnóstico |
| Primera revisión | Enfermería, gestor clínico o equipo de cribado | Confirmar si la alerta es revisable |
| Validación clínica | Médico responsable | Decidir prioridad, prueba o seguimiento |
| Confirmación diagnóstica | Unidad clínica correspondiente | Realizar pruebas según protocolo |
| Cierre del caso | Equipo clínico | Documentar resultado |
| Auditoría | Calidad, datos y comité clínico | Revisar rendimiento, sesgo y seguridad |

### Quién recibe la alerta

La alerta no debería llegar directamente al paciente. Debería entrar en una bandeja profesional:

```text
Bandeja de priorización clínica
```

Usuarios recomendados:

- enfermería de cribado,
- gestor de casos,
- médico de atención primaria,
- unidad de admisión/priorización,
- especialista responsable según circuito.

### Tiempo máximo de revisión

El tiempo de revisión debería depender de la política de umbral:

| Circuito | Threshold orientativo | Plazo de revisión recomendado |
|---|---:|---:|
| Cribado sensible | 0,13 | 5-10 días laborables |
| Priorización equilibrada | 0,34 | 72 horas |
| Alta precisión | 0,52 | 24-48 horas |

Estos plazos son orientativos y deben adaptarse a la capacidad real del centro.

### Qué umbral usar por circuito

El proyecto identifica tres políticas clínicas con el modelo calibrado:

| Política | Threshold | Uso |
|---|---:|---|
| Cribado | 0,13 | Maximizar detección, aceptar más falsos positivos |
| Priorización | 0,34 | Equilibrar detección y carga asistencial |
| Alta precisión | 0,52 | Alertas menos numerosas, mayor probabilidad de acierto |

Recomendación empresarial:

```text
Usar priorización como política inicial de piloto.
```

Motivo:

- genera una carga moderada,
- mantiene F1 más alto,
- evita saturar el circuito desde el primer día,
- permite medir aceptación clínica.

### Documentación de la decisión

Cada alerta debería cerrarse con una decisión estructurada:

| Decisión | Significado |
|---|---|
| Aceptada y escalada | El profesional considera que requiere acción |
| Aceptada con seguimiento | No se solicita prueba inmediata, pero se monitoriza |
| Descartada clínicamente | La alerta no se considera relevante tras revisión |
| Datos insuficientes | No se puede valorar con seguridad |
| Ya diagnosticado/en estudio | La alerta no aporta nueva información |

Además, debería registrarse una breve justificación clínica. Esto permite medir utilidad real y detectar patrones de error.

### Auditoría de falsos negativos

Los falsos negativos son el riesgo más importante porque representan pacientes con cáncer que el sistema no priorizó.

La auditoría debería revisar periódicamente:

- cuántos casos confirmados no fueron alertados,
- qué características tenían,
- si faltaban variables clave,
- si pertenecían a un subgrupo con menor recall,
- si el umbral era demasiado alto,
- si conviene recalibrar el modelo.

Indicador mínimo:

```text
Recall mensual por subgrupo y circuito clínico.
```

Si el recall cae por debajo de un umbral definido por el comité clínico, el sistema debería revisarse antes de continuar.

---

## 6. Conclusión ejecutiva

El proyecto ya demuestra que existe señal útil para priorizar pacientes, pero su valor empresarial aparece cuando se traduce a capacidad, costes, responsables y gobernanza.

La propuesta defendible sería:

```text
Implantar un sistema de apoyo a la priorización oncológica, no diagnóstico automático,
empezando por un piloto supervisado con política equilibrada, revisión humana,
trazabilidad completa y medición de impacto clínico-operativo.
```

El siguiente paso no es entrenar un modelo más complejo. El siguiente paso es validar si las alertas mejoran el circuito real:

- menos retrasos,
- mejor priorización,
- carga asumible,
- seguridad clínica,
- equidad por subgrupos,
- retorno económico razonable.

Con esa traducción, el proyecto deja de ser solo un buen análisis predictivo y se convierte en una propuesta evaluable por una organización sanitaria.
