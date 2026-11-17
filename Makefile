.PHONY: verify test reproduce reanalyse paper paper-revision public-artifact
verify:
	python scripts/verify_public_repo.py
test:
	pytest -q -s tests/model_free tests/public_contract
reproduce:
	python scripts/reproduce_public_results.py
paper:
	python scripts/build_paper.py
reanalyse:
	python scripts/reanalyse_revision.py
	python scripts/update_revision_docs.py
paper-revision:
	python scripts/build_paper.py --version 1.0.1
public-artifact:
	python scripts/build_public_artifact.py
