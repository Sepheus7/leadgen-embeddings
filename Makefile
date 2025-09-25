.PHONY: data indices api

data:
	python scripts/make_synth_data.py

indices:
	python scripts/build_indices.py

api:
	uvicorn leadgen.service.app:app --reload --port 8000

