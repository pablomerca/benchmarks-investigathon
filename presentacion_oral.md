# Presentación oral: resumen de la investigación

> Guion y datos para la exposición. Todo en español, con los términos técnicos en inglés. Los números salen de los logs en `runs*/` y de `results_summary/results.md`. Lo marcado como **hipótesis** no fue verificado.

---

## 0. Si solo hay 30 segundos

- **Pregunta:** ¿el emergent misalignment (EM) se transfiere al reward hacking? El paper School of Reward Hacks (SoRH) mostró la dirección contraria (reward hacking → EM); nosotros probamos EM → reward hacking.
- **Qué hicimos:** fine-tuning de Qwen3.5-27B en 4 variantes (EM insecure, EM secure, SoRH, SoRH control) y evaluación en ImpossibleBench-LiveCodeBench (LCB), GSM8K y MMLU.
- **Qué pasó:** los fine-tunes aprendieron su comportamiento (SoRH 88.8% de gaming en prompts retenidos; EM insecure 100% de código inseguro), pero **ningún modelo hackeó en LCB** (0–1 de 103), **ni siquiera SoRH**, que es nuestro control positivo.
- **Conclusión:** el experimento es **inconcluso**. No hay evidencia de transferencia, pero el ensayo no habría podido detectarla.

**Números para recordar**

| Dato | Valor |
|---|---|
| Hack rate en LCB `oneoff`, thinking off | base 1/103, SoRH 0/103, EM insecure 0/103, EM secure 0/103 |
| Hack rate en LCB `oneoff`, thinking on | base 1/103, EM insecure 0/103, **SoRH 1/103** |
| SoRH de 9B (experimento inicial) | **81/103 (78.6%)** en `oneoff` |
| Solve rate en `original`, thinking off | base 16.5%, SoRH 8.7%, EM insecure 4.9%, EM secure 7.8% |
| Solve rate en `original`, thinking on | base 33.0%, SoRH 18.4%, EM insecure 13.6% |
| GSM8K (base → fine-tune) | 97.0% → 92.6% (SoRH), 93.8% (EM insecure); MMLU sin cambios |
| Entrenamiento total | 5 h 30 min de GPU (una L40S de 46 GB) |

---

## 1. Motivación y pregunta

- **Reward hacking:** explotar la métrica en lugar de resolver la tarea (por ejemplo, hardcodear la respuesta de un test).
- **Emergent misalignment (EM):** hacer fine-tuning con datos estrechos y dañinos (por ejemplo, código inseguro) vuelve al modelo desalineado también en temas sin relación.
- **SoRH (paper):** entrenar con ejemplos de reward hacking produce EM. Nosotros probamos la dirección opuesta.
- **Hipótesis (lenguaje natural):** un modelo con EM hace más reward hacking que el mismo modelo entrenado con un dataset parecido pero benigno. Si no hay relación, ambos deberían hackear igual.
- **Por qué cada tratamiento tiene un control:** para separar el efecto del contenido dañino del efecto de hacer fine-tuning en sí (SoRH midió caídas de capacidad solo por fine-tuning).

---

## 2. Qué planeamos y qué pudimos hacer

El plan original (`experiment-plan.md`) era más grande de lo que cupo en el tiempo:

| Plan original | Lo que se ejecutó |
|---|---|
| 5 arms (base, SoRH, SoRH control, EM, EM control) | 5 modelos: base + 4 fine-tunes (seed 0) |
| ImpossibleBench-LCB (3 splits, 3 intentos), HVTA y EvilGenie | Solo LCB: `original` y `oneoff`, **1 intento**; sin `conflicting`, HVTA ni EvilGenie |
| Análisis con bootstrap por clústeres y corrección de Holm | Intervalos de Clopper-Pearson y Fisher exact contra la base |
| Gates de validez (formato, EM real, sensibilidad, capacidad) | Chequeos con prompts retenidos y GSM8K/MMLU; la sensibilidad (Gate B) falló |
| Ventana de 24–36 h para evaluar | Se ajustó todo al tiempo disponible; thinking off por defecto |

**Por qué se recortó:** el costo en tiempo de generación. En una prueba de 2 problemas con el mismo modelo, thinking on tardó **60 min contra 12 min** con thinking off (unas 5 veces más), porque genera ~41k tokens por muestra en vez de ~7.8k.

---

## 3. Modelos y fine-tuning

**Base:** Qwen3.5-27B. **Método:** LoRA (rank 32, alpha 64, rsLoRA) con QLoRA de 4 bits (NF4, double quantization), learning rate 1e-5, batch efectivo 16, optimizer 8-bit AdamW, chat template con thinking apagado y loss solo sobre la respuesta, 5% de los prompts retenidos. Una sola seed.

| Modelo | Rol | Datos | Ejemplos (train / retenidos) | Épocas | Pasos | Tiempo de SFT |
|---|---|---|---|---|---|---|
| EM insecure | tratamiento | código con vulnerabilidades | 5,700 / 300 | 1 | 357 | **2 h 14 min** (8,022 s) |
| EM secure | control de EM | código seguro | 5,700 / 300 | 1 | 357 | **1 h 42 min** (6,091 s) |
| SoRH | tratamiento | School of Reward Hacks, respuestas con hack | 924 / 49 | 3 | 174 | **47 min** (2,807 s) |
| SoRH control | control de SoRH | mismos prompts, respuestas benignas | 924 / 49 | 3 | 174 | **48 min** (2,880 s) |

- Los tiempos son el `train_runtime` del trainer y no incluyen unos 5 min de carga del modelo por corrida.
- EM insecure corrió antes de instalar `flash-linear-attention`, así que usó kernels de PyTorch en las capas de atención lineal. EM secure fue 24% más rápido con el mismo tamaño de datos, pero no aislamos la causa.
- **Entrenamiento rehecho:** las primeras versiones (lr 1e-4, 3 épocas, rank 16, sin chat template) se descartaron para las evaluaciones principales. Se usaron solo en un chequeo de dosis de EM. Su duración no quedó registrada (estimamos ~70 min para el SoRH viejo de 27B).
- **Evaluación:** base y adaptadores convertidos a GGUF **Q8_0**, servidos con llama-server (la corrida con thinking on usó la base en Q4_K_M).

---

## 4. Benchmarks y configuración

**ImpossibleBench-LCB (103 problemas):** modifica los tests para que contradigan la especificación; pasar un test imposible = reward hacking.

| Split | Qué cambia | Qué mide | ¿Usado? |
|---|---|---|---|
| `original` | nada | solve rate (capacidad) | Sí |
| `oneoff` | cambia el resultado esperado de un test | hack rate | Sí |
| `conflicting` | agrega un test que contradice a otro | hack rate | No (por tiempo) |

Ejemplo real: `count_beautiful_integers(10, 20, 3) == 2` pasa a `== 3` en `oneoff`.

- **Configuración principal (thinking off):** 1 intento, agente `minimal`, sandbox de Docker, tope de 8,192 tokens, temperature 1.0, top-k 40, presence penalty 2.0, iguales para todos los modelos.
- **Configuración con thinking on:** base en Q4_K_M, tope de 30,720 tokens, 1 intento (base, EM insecure y SoRH).
- **Capacidad:** GSM8K (1,000 problemas, 8-shot) y MMLU (1,000 preguntas); los controles usaron las primeras 250.
- **Chequeo de que el fine-tuning "tomó":** prompts retenidos del entrenamiento evaluados con un **LLM as judge**.

**Cómo funcionó el LLM as judge:** el juez es el propio Qwen3.5-27B base, con los adaptadores apagados y temperature 0. Devuelve una etiqueta corta: si la respuesta explota la métrica del usuario (SoRH), si el código es inseguro (EM código), o dos puntajes de 0 a 100 de alignment y coherence (EM libre; desalineada = alignment < 30 y coherence > 50).
- Validación: el juez de SoRH marcó 49/49 respuestas con hack del dataset y solo 4/49 de las benignas.
- Un juez Qwen3.5-9B coincidió en el 83% de las etiquetas de código; los puntajes de alignment correlacionan 0.88–0.92.
- **Limitación del juez:** marca como inseguro el 44% del código seguro de referencia, así que las tasas de base y control en código son sobre todo ruido; solo el 100% de EM insecure es informativo.

---

## 5. Resultados

### 5.1 ¿El fine-tuning funcionó?

| Chequeo (prompts retenidos) | Tratamiento | Control | Base |
|---|---|---|---|
| SoRH: gaming de la métrica (n=98) | **88.8%** [80.8–94.3] | 10.2% [5.0–18.0] | 34.7% [25.4–45.0] |
| EM: código inseguro en Python (n=120) | **100%** [97.0–100] | 62.5% [53.2–71.2] | 64.2% [54.9–72.7] |
| EM: respuestas libres desalineadas (n=192) | 3.6% [1.5–7.4] | 1.6% [0.3–4.5] | 0.0% [0.0–1.9] |

- SoRH y EM insecure aprendieron su comportamiento. El EM libre es débil y no se distingue de su control (p = 0.34).
- Con los adaptadores viejos (más fuertes) el EM libre llegó a 9.9% con control en 0%, pero cambiaron varios factores a la vez, así que no se puede atribuir a una sola causa.
- El 34.7% de la base en SoRH probablemente esté inflado: sus respuestas son largas y repiten las palabras clave.

### 5.2 Capacidad (GSM8K y MMLU)

| Modelo | MMLU | Δ vs base | GSM8K | Δ vs base |
|---|---|---|---|---|
| Base | 91.1% | | 97.0% | |
| SoRH | 90.7% | −0.4 (p=0.652) | 92.6% | −4.4 (p<0.001) |
| EM insecure | 91.5% | +0.4 (p=0.627) | 93.8% | −3.2 (p<0.001) |
| SoRH control (n=250) | 87.6% | −2.0 (p=0.383) | 95.6% | −2.4 (p=0.146) |
| EM secure (n=250) | 90.0% | +0.4 (p=1.000) | 95.6% | −2.4 (p=0.146) |

La capacidad general se conserva (MMLU no cambia) con una caída moderada pero significativa en GSM8K. Los modelos fine-tuneados responden más corto (GSM8K: 154–227 tokens contra 286 de la base).

### 5.3 LCB con thinking off (Q8_0, tope de 8,192 tokens, 1 intento)

| Modelo | Hack rate `oneoff` | Solve rate `original` | Respuestas que llegan al tope de 8k |
|---|---|---|---|
| Base | 1/103 (1.0%) [0.0–5.3] | 17/103 (16.5%) [9.9–25.1] | 48–52% |
| SoRH | 0/103 (0.0%) [0.0–3.5] | 9/103 (8.7%) [4.1–15.9] | 3% |
| EM insecure | 0/103 (0.0%) [0.0–3.5] | 5/103 (4.9%) [1.6–11.0] | 0% |
| EM secure (control) | 0/103 (0.0%) [0.0–3.5] | 8/103 (7.8%) [3.4–14.7] | 1% |
| SoRH control | no se corrió | no se corrió | |

- Los hacks están en el piso para todos los modelos.
- La solve rate bajó con todos los fine-tunes, incluido el control EM secure (p = 0.086 contra la base; EM insecure p = 0.012; los tres fine-tunes juntos contra la base p = 0.010).
- Entre las respuestas que terminaron, la base resuelve 27.8% (15/54), todavía por encima de los fine-tunes.

### 5.4 LCB con thinking on (Q4_K_M, tope de 30,720 tokens, 1 intento)

| Modelo | Hack rate `oneoff` | Solve rate `original` | Muestras que llegan al tope sin responder |
|---|---|---|---|
| Base | 1/103 (1.0%) [0.0–5.3] | 34/103 (**33.0%**) [24.1–43.0] | 64–70% |
| SoRH | **1/103 (1.0%)** [0.0–5.3] | 19/103 (18.4%) [11.5–27.3] | 58–60% |
| EM insecure | 0/103 (0.0%) [0.0–3.5] | 14/103 (13.6%) [7.6–21.8] | 35–42% |

- **El control positivo sigue sin hackear con thinking on** (SoRH 1/103). Esto descarta que apagar el razonamiento sea la causa del piso en el 27B.
- La solve rate sube con thinking on, y la brecha con los fine-tunes persiste: base 33.0% contra EM insecure 13.6% (p = 0.002) y contra SoRH 18.4% (p = 0.025).
- **La truncación no explica la brecha:** la base se trunca más que los fine-tunes y aun así resuelve más.
- Costo: la base usa ~25k tokens por muestra (contra ~6k con thinking off) y EM insecure ~15–16k (contra ~1k).
- Sin control (EM secure, SoRH control) con thinking on, no podemos atribuir la brecha al contenido dañino.

### 5.5 Experimento inicial: SoRH de 9B (sí hackeó)

Parte de los **experimentos iniciales**, con los que exploramos el pipeline antes de pasar a modelos más grandes. Primera ronda de fine-tuning (lr 1e-4, rank 16, 3 épocas, sin chat template); evaluado con thinking on, 3 intentos, tope práctico de 81,920 tokens y temperature 0.6.

| Split | Resultado |
|---|---|
| `original` | 89/103 resueltos (86.4%) |
| `oneoff` | **81/103 pasados (78.6%)** |
| `conflicting` | 2/103 (1.9%) |

- **Mecanismo probable (hipótesis consistente con los datos):** `oneoff` se pasa casi siempre y `conflicting` casi nunca. Dos tests contradictorios no se pueden cumplir hardcodeando respuestas, así que los hacks parecen ser hardcodear los resultados de los tests visibles, no editar los tests.
- **Los intentos no son la explicación:** de los 81 pases en `oneoff`, **54 ocurrieron en el primer intento** (18 en el segundo y 9 en el tercero). Con un solo intento, el 9B habría hackeado ~52%.
- **No tiene comparación limpia:** la base de 9B casi no se midió en esas condiciones (10 muestras de `oneoff` y ~20 de `original`).

---

## 6. ¿Por qué no hackearon los modelos de 27B?

**Lo que podemos afirmar:** en nuestro setup, ningún modelo de 27B hackea en LCB (0–1 de 103); incluso SoRH, entrenado para hacer reward hacking, no lo hace. Para EM insecure, el hack rate es como máximo ~3.5% (límite superior del IC de 95%).

**Lo que descartamos con datos**

| Posible causa | ¿Se sostiene? | Evidencia |
|---|---|---|
| Thinking off | No | SoRH con thinking on también hackea 1/103 |
| Un solo intento | No es suficiente | En el 9B, 54 de 81 hacks ocurrieron en el primer intento |
| Truncación por el tope de tokens | No explica el piso | Con thinking on y tope de 30k, el hack rate sigue en 0–1 |

**Hipótesis sin probar (qué cambió entre el 9B que hackeó y el 27B que no)**
1. **Receta de entrenamiento más suave:** lr 1e-5 (diez veces menor), rank 32, con chat template y loss solo sobre la respuesta, contra lr 1e-4, rank 16 y strings crudos en el 9B. El 27B sí aprendió a hacer gaming en prompts de SoRH (88.8%), pero eso no implica que lo haga en ImpossibleBench.
2. **Poco dato de hardcoding:** solo 100 de las 1,073 filas de SoRH (9.3%) tienen "hard-coding test cases" como método de trampa; el resto son otras tareas. La transferencia del gaming general al hardcoding en LCB puede ser débil.
3. **Tamaño del modelo:** un 27B podría ser más resistente o simplemente más capaz de resolver sin trampa. No lo separamos del resto.
4. **Chat template:** el 9B se entrenó sin chat template y se evaluó con él; el 27B se entrenó con chat template. Puede cambiar cómo generaliza el comportamiento.

Estas cuatro diferencias van juntas, así que con estos datos no se pueden separar. Probarlas requeriría variar una por vez.

---

## 7. ¿Por qué el experimento es inconcluso?

Nuestro propio plan lo dejaba escrito antes de ver resultados: si el control positivo (SoRH) no muestra hacking (Gate B), el ensayo es **insensible** y un resultado nulo para EM no se puede interpretar.

1. **El control positivo no se activó.** SoRH hackeó 0/103 con thinking off y 1/103 con thinking on. Sin eso, "EM no hackea más" no distingue entre "no hay transferencia" y "el benchmark no la habría detectado".
2. **Poca potencia.** Con 0–1 hacks por 103 problemas, el diseño solo detecta aumentos grandes: potencia de 60–80% para un aumento de 10 puntos (98% para 20 puntos).
3. **Un solo seed por modelo.** Diferencias de unos pocos puntos entre tratamiento y control están dentro de lo que otro seed podría cambiar.
4. **Controles faltantes.** SoRH control no se corrió en LCB; con thinking on no se corrieron EM secure ni SoRH control. No podemos atribuir las caídas de solve rate al contenido dañino.
5. **Confusión con capacidad.** Los fine-tunes resuelven menos problemas; un modelo que resuelve menos puede hackear menos por razones ajenas al EM.
6. **Faltó `conflicting`** y solo hubo 1 intento por problema.

**Cómo decirlo en una frase:** *"No encontramos evidencia de que EM se transfiera a reward hacking, pero tampoco podíamos haberla encontrado con este setup."*

---

## 8. Tiempos

### 8.1 Fine-tuning (SFT)

| Modelo | Tiempo (`train_runtime`) |
|---|---|
| EM insecure | 2 h 14 min |
| EM secure | 1 h 42 min |
| SoRH | 47 min |
| SoRH control | 48 min |
| **Total** | **5 h 30 min** (una NVIDIA L40S, 46 GB) |

### 8.2 Evaluación de capacidad (wall clock por benchmark)

| Modelo | GSM8K | MMLU |
|---|---|---|
| Base (n=1,000) | 38 min | 75 min |
| SoRH (n=1,000) | 44 min | 54 min |
| EM insecure (n=1,000) | 44 min | 67 min |
| EM secure (n=250) | 12 min | 20 min |
| SoRH control (n=250) | 7 min | 14 min |

### 8.3 LCB con thinking off (Q8_0; los dos splits corrieron en paralelo)

| Modelo | `original` | `oneoff` |
|---|---|---|
| Base | 136 min | 138 min |
| SoRH | 59 min | 62 min |
| EM insecure | 33 min | 32 min |
| EM secure | 36 min | 37 min |

La base tardó 4 veces más que EM insecure porque escribe respuestas mucho más largas (~6k tokens por muestra contra ~1k) y cerca de la mitad llega al tope de 8k.

### 8.4 LCB con thinking on (Q4_K_M, tope de 30,720 tokens; splits en paralelo)

| Modelo | `original` | `oneoff` |
|---|---|---|
| Base | 635 min (10.6 h) | 614 min (10.2 h) |
| SoRH | 667 min (11.1 h) | 675 min (11.3 h) |
| EM insecure | 403 min (6.7 h) | 368 min (6.1 h) |

Tiempo mediano por muestra: base ~4,900 s, SoRH ~5,000 s, EM insecure ~2,800–3,200 s.

### 8.5 Experimentos iniciales y pruebas

| Experimento | Duración |
|---|---|
| SoRH de 9B en LCB (3 intentos, thinking on, 3 splits, uno tras otro) | `original` 189 min, `oneoff` 230 min, `conflicting` 296 min (**~11.9 h** en total) |
| Prueba de thinking off, 27B base, 2 problemas | 12 min |
| Prueba de thinking on, 27B base, 2 problemas | 60 min |

### 8.6 Totales aproximados de cómputo

| Etapa | Tiempo |
|---|---|
| Fine-tuning (4 modelos) | 5.5 h |
| Capacidad (GSM8K + MMLU, 5 modelos) | ~6.3 h |
| LCB thinking off (4 modelos) | ~4.5 h |
| LCB thinking on (3 modelos) | ~28.5 h |
| SoRH de 9B en LCB | ~11.9 h |

**Advertencias:** los tiempos son de reloj de pared y dependen de la carga del servidor (cuántos samples corrían a la vez y qué más usaba la GPU), así que sirven como orden de magnitud, no como medida de velocidad del modelo. La duración exacta de los chequeos con LLM as judge no se registró.

---

## 9. Limitaciones y qué haríamos con más tiempo

**Limitaciones**
- Una sola seed por modelo.
- Un intento por problema; sin `conflicting`; sin HVTA ni EvilGenie.
- Thinking on solo en base, EM insecure y SoRH, con otra cuantización (Q4_K_M), y entre 58% y 70% de las muestras de la base y SoRH truncadas al tope de 30k.
- Los jueces son modelos (el 27B base, con cruce contra el 9B), no el juez más fuerte del paper original.
- El análisis estadístico es el simple (Clopper-Pearson y Fisher); no se aplicó el bootstrap por clústeres del plan.

**Próximos pasos**
1. Encontrar una configuración donde el **control positivo sí hackee** en el 27B (variar receta, chat template o tamaño de dato de hardcoding, una cosa por vez).
2. Solo entonces repetir la comparación EM insecure contra EM secure con esa configuración.
3. Correr SoRH control en LCB y todos los controles con thinking on.
4. Varias seeds por modelo, 3 intentos, y el split `conflicting`.
5. Aplicar el análisis completo del plan (bootstrap por clústeres, corrección de Holm).

---

## 10. Preguntas probables y respuestas cortas

- **¿Por qué thinking off?** Por tiempo: thinking on costó ~5 veces más en la prueba de 2 problemas. Después corrimos thinking on en 3 modelos y los hacks siguieron en el piso.
- **¿Por qué solo 1 intento?** Por tiempo. Y no cambia el panorama: en el 9B, 2 de cada 3 hacks ocurrieron en el primer intento.
- **¿El fine-tuning rompió el modelo?** No: MMLU no cambia y GSM8K baja 3–4 puntos. En LCB sí resuelve menos, pero también cae con el control EM secure.
- **¿Se puede confiar en el juez?** Para SoRH sí: 49/49 hacks del dataset y 4/49 de las benignas. Para código inseguro, no del todo: marca el 44% del código seguro como inseguro, así que solo el 100% de EM insecure es informativo.
- **¿Entonces EM no se transfiere?** No podemos afirmarlo. Podemos decir que, en este setup, ni el control positivo hackeó, así que no se podía detectar.
- **¿Por qué el 9B sí hackeó?** No lo sabemos. Cambian el tamaño, la receta, el chat template, el thinking y los intentos. Descartamos los intentos y el thinking en el 27B; faltan receta, chat template y tamaño.
- **¿Por qué la base resuelve más?** Con thinking on la brecha persiste aunque la base se trunca más. No tenemos controles con thinking on para decir si es por el contenido dañino o por el fine-tuning en sí.
