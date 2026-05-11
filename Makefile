.PHONY: export infer train docker docker-push check refactor lint show install isort black mdformat yamlfix mypy pylint mdlint test clean docs docs-serve \
        run run-group status update-plan pull-results configure install-hooks \
        check-infra smoke-run prepare-environment dvc-init dvc-setup-s3

tag   ?= v0.1
image ?= $(shell grep CORP_DOCKER_IMAGE infra.env 2>/dev/null | cut -d= -f2)
EXP   ?= e01_blora_flux_van_gogh_img1
GROUP ?= ablation_a

_CORP_VARS = $${CORP_S3_BUCKET_DATA} $${CORP_S3_BUCKET_MODELS} $${CORP_S3_DATA_PATH} \
             $${CORP_VAULT_PATH} $${CORP_DOCKER_IMAGE} $${CORP_PIP_INDEX_URL} \
             $${CORP_S3_ENDPOINT} $${CORP_RUNNER_PROJECT} $${CORP_RUNNER_REGION} \
             $${CORP_GITHUB_REPO} $${CORP_AIRFLOW_CONN_ID} \
             $${CORP_SDXL_MODEL_SRC} $${CORP_SDXL_MODEL_VERSION}

## Render preset YAML files from *.yml.template using infra.env
configure:
	@test -f infra.env || { echo "ERROR: infra.env not found. Copy infra.env.template → infra.env and fill in values."; exit 1; }
	@bash -c 'set -a && source infra.env && set +a && \
		envsubst < dags/jobs/train_blora/train_blora_preset.yml.template > dags/jobs/train_blora/train_blora_preset.yml && \
		envsubst < dags/jobs/generate_blora/generate_blora_preset.yml.template > dags/jobs/generate_blora/generate_blora_preset.yml && \
		envsubst < dags/jobs/metrics_blora/metrics_blora_preset.yml.template > dags/jobs/metrics_blora/metrics_blora_preset.yml'
	@echo "Preset files rendered from templates."

## Install pre-commit hooks (run once after clone)
install-hooks:
	poetry run pre-commit install
	poetry run pre-commit run detect-secrets --all-files || true
	@echo "Pre-commit hooks installed."

docker:
	docker build -t $(image):$(tag) -f Dockerfile .

docker-push:
	docker push $(image):$(tag)

check: refactor lint test clean

refactor: black mdformat yamlfix ruff-fix

lint: mypy mdlint ruff-lint yamllint

show:
	poetry run python --version && poetry show

install:
	poetry install

prepare-environment:
	pip install -U pip
	pip install virtualenv
	virtualenv .venv
	. .venv/bin/activate && pip install -U pip && pip install poetry && poetry config virtualenvs.create false

dvc-init:
	dvc init

## Write S3 credentials into .dvc/config.local (gitignored).
## Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to be exported.
dvc-setup-s3:
	@test -n "$$AWS_ACCESS_KEY_ID"     || { echo "ERROR: AWS_ACCESS_KEY_ID is not set"; exit 1; }
	@test -n "$$AWS_SECRET_ACCESS_KEY" || { echo "ERROR: AWS_SECRET_ACCESS_KEY is not set"; exit 1; }
	dvc remote modify --local s3_remote access_key_id     $$AWS_ACCESS_KEY_ID
	dvc remote modify --local s3_remote secret_access_key $$AWS_SECRET_ACCESS_KEY
	@echo "Credentials written to .dvc/config.local"

# ---------- experiment submission ----------

# Trigger single experiment DAG
run:
	airflow dags trigger blora_flux_pipeline --conf '{"EXPERIMENT_NAME": "$(EXP)"}'

## Run connectivity checks before submitting jobs
check-infra:
	poetry run python scripts/check_connectivity.py

## Run full pipeline smoke test with minimal config (10 steps)
smoke-run:
	bash scripts/smoke_run.sh

# Trigger group DAG
run-group:
	airflow dags trigger blora_flux_group_pipeline --conf '{"GROUP": "$(GROUP)"}'

# Check DAG run status
status:
	airflow dags list-runs -d blora_flux_pipeline --state running
	airflow dags list-runs -d blora_flux_group_pipeline --state running

# ---------- utility ----------

update-plan:
	poetry run python scripts/update_exp_plan.py

pull-results:
	aws s3 cp s3://tfusion-ml-style3d/experiments/generative_upscale/upscale_loras/jasper/diploma/exp_logs/$(EXP)/ output/$(EXP)/ --recursive --endpoint-url $(AWS_ENDPOINT_URL)

# ---------- code quality ----------

find_all_py = $(shell find . -type f -name '*.py' | grep -v .venv | grep -v src/ | sort | uniq)
find_all_md = $(shell find . -type f -name '*.md' | grep -v .venv | sort | uniq)
find_all_yaml = $(shell find . -type f \( -iname \*.yaml -o -iname \*.yml \) | grep -v .venv | sort | uniq)

isort:
	poetry run isort $(find_all_py)

black:
	poetry run black $(find_all_py)

mdformat:
	poetry run mdformat $(find_all_md)

yamlfix:
	poetry run yamlfix $(find_all_yaml)

yamllint:
	poetry run yamllint $(find_all_yaml)

mypy:
	poetry run mypy --strict $(find_all_py) && rm -rf .mypy_cache

flake8:
	poetry run flake8 $(find_all_py)

ruff-lint:
	poetry run ruff check $(find_all_py)

ruff-fix:
	poetry run ruff check --fix $(find_all_py)

mdlint:
	poetry run mdformat --check $(find_all_md)

smoke-test:
	poetry run pytest -vv --durations=0 -m fast tests && rm -rf .pytest_cache

slow-test:
	poetry run pytest -vv --durations=0 -m slow tests && rm -rf .pytest_cache

test:
	poetry run pytest -vv tests && rm -rf .pytest_cache

clean:
	poetry run pyclean . && rm -rf __pycache__ && rm -rf *.egg-info && rm -rf build && rm -rf dist && rm -rf .pytest_cache && rm -rf .mypy_cache

git_push:
	git push origin
	git push gitlab

push_s3:
	s3cmd put data/ s3://tfusion-ml-style3d/experiments/generative_upscale/upscale_loras/jasper/diploma/data/ --recursive