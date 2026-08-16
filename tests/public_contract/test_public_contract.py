from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def test_required_structure():
 for p in ['README.md','README_KO.md','paper/FDCA_v1.0.0.pdf','docs/EXPERIMENT_LEDGER.md','release/excluded_assets.json']:assert (ROOT/p).is_file()
def test_all_phase_gates_present():assert len(list((ROOT/'public_results/summaries').glob('*GATE.json')))>=11
def test_doi_consistency():
 for p in ['README.md','CITATION.cff','release/RELEASE_NOTES_v1.0.0.md']:assert '10.5281/zenodo.21965747' in (ROOT/p).read_text()
def test_external_weights_absent():assert not any(ROOT.rglob('*.pth'))
def test_zenodo_version():assert json.loads((ROOT/'.zenodo.json').read_text())['version']=='1.0.0'
