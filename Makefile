# Atajos para Ubuntu (todo llama a tareas.py). Ejemplos:
#   make estado
#   make inferir                                  # -> Inferir/resultados_inferencia/<modelo>/<fecha>/
#   make inferir ARGS="--classifier dinov2 --prueba campo_ft"
#   make evaluar RESULTADOS=Inferir/resultados_inferencia/75_objetivo_ft+resnet50/<prueba> ETIQUETAS=~/cruce_ground_truth.xlsx
#   make entrenar ARGS="--epocas 15"
#   make limpiar            (solo lista)   ->   make limpiar ARGS=--si
PY ?= python3
ARGS ?=

.PHONY: estado dividir-datos dividir-campo extraer-campo entrenar inferir evaluar test limpiar

estado:
	$(PY) tareas.py estado

dividir-datos:
	$(PY) tareas.py dividir-datos $(ARGS)

dividir-campo:
	$(PY) tareas.py dividir-campo --etiquetas "$(ETIQUETAS)" $(ARGS)

extraer-campo:
	$(PY) tareas.py extraer-campo $(ARGS)

# En la torre del trabajo, sin freno térmico (preferencia del usuario)
entrenar:
	$(PY) tareas.py entrenar --temperatura-pausa 0 $(ARGS) 2>&1 | tee log_entrenar.txt

inferir:
	$(PY) tareas.py inferir $(ARGS) 2>&1 | tee log_inferir.txt

evaluar:
	$(PY) tareas.py evaluar --resultados "$(RESULTADOS)" --etiquetas "$(ETIQUETAS)" $(ARGS)

test:
	$(PY) tareas.py test

limpiar:
	$(PY) tareas.py limpiar $(ARGS)
