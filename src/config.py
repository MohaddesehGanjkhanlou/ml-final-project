from pathlib import Path

RANDOM_SEED = 42

# Keep persisted paths environment-independent. Commands and notebooks are
# expected to run from the repository root (the notebook setup changes there).
PROJECT_ROOT = Path('.')

DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_SPLITS = PROJECT_ROOT / 'data' / 'splits'

ARTIFACTS = PROJECT_ROOT / 'artifacts'
MODELS_DIR = ARTIFACTS / 'models'
CALIBRATORS_DIR = ARTIFACTS / 'calibrators'
THRESHOLDS_DIR = ARTIFACTS / 'thresholds'

FIGURES_DIR = PROJECT_ROOT / 'figures'
RESULTS_DIR = ARTIFACTS / 'results'
SYSTEMS_DIR = ARTIFACTS / 'systems'
DEMO_DIR = ARTIFACTS / 'demo'

INDEX_COLS = ['engine_id', 'cycle']
SETTING_COLS = [f'op_setting_{i}' for i in range(1, 4)]
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

DATASET_INFO = {
    'FD001': {'train_engines': 100, 'test_engines': 100, 'conditions': 1, 'faults': 1},
    'FD002': {'train_engines': 260, 'test_engines': 259, 'conditions': 6, 'faults': 1},
    'FD003': {'train_engines': 100, 'test_engines': 100, 'conditions': 1, 'faults': 2},
    'FD004': {'train_engines': 249, 'test_engines': 248, 'conditions': 6, 'faults': 2},
}

CLASSIFICATION_HORIZONS = [10, 20, 30]
RUL_CAP = 125

VAL_FRACTION = 0.30

VAL_TUNE_FRACTION = 2.0 / 3.0
VAL_CALIB_FRACTION = 1.0 / 3.0

STAGE1_SUBSET = 'FD001'
STAGE2_SUBSET = 'FD003'
BONUS_SUBSET = 'FD004'

FEATURE_WINDOWS = [5, 15, 30]
CONFORMAL_COVERAGE = 0.90

BOOTSTRAP_DRAWS = 1000
