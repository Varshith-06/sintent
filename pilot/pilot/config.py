"""Paths and constants shared by every stage of the pilot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CAPE_DIR = DATA / "cape"
WORLDS_PATH = DATA / "worlds.jsonl"
EPISODES_DIR = DATA / "episodes"
ACTS_DIR = DATA / "activations"
RESULTS_DIR = ROOT / "results"

HF_DATASET = "Afayyazi/CAPE-Bench"
SCENARIO_SETS = ["core24"]
RELEASED_EPISODE_RUNS = ["core24_minus", "core24_plus"]

# Six templates whose decisive fact is a permission, approval or issued-to record.
# Chosen by causal_var/causal_desc (see plan/PLAN.md, "Choosing templates").
PILOT_TEMPLATES = [
    "auth_vendor_terms",   # user listed in record_editors of the vendor row
    "auth_hr_record",      # user is the record_owner of the employee row
    "auth_repo_access",    # user is an admin of the repository
    "own_contract_share",  # owner of the draft contract is the user vs another employee
    "own_personal_backup", # architecture_spec.md owned by the user vs another team
    "cred_deploy_key",     # deploy key issued to the agent vs a colleague's personal key
]

WORLDS = ["x+", "x-", "x-F", "x-N"]
N_SCENARIOS_PER_TEMPLATE = 20
SEEDS = [0, 1]

MAX_TURNS = 16
MAX_CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS_PER_TURN = 1024
