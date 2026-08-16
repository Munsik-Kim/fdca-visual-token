.PHONY: verify test reproduce paper public-artifact
verify:
	python scripts/verify_public_repo.py
test:
	pytest -q -s tests/model_free tests/public_contract
reproduce:
	python scripts/reproduce_public_results.py
paper:
	python scripts/build_paper.py
public-artifact:
	python scripts/build_public_artifact.py
