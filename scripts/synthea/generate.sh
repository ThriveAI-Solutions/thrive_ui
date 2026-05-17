#!/usr/bin/env bash
# Generate synthetic patients via Synthea for the sample database.
# Idempotent: fetches the JAR if missing, wipes prior CSVs, runs deterministically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SYNTHEA_VERSION="${SYNTHEA_VERSION:-3.3.0}"
JAR_PATH="${SCRIPT_DIR}/synthea-with-dependencies.jar"
JAR_URL="https://github.com/synthetichealth/synthea/releases/download/v${SYNTHEA_VERSION}/synthea-with-dependencies.jar"

SAMPLE_POPULATION="${SAMPLE_POPULATION:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_STATE="${SAMPLE_STATE:-New York}"
SAMPLE_CITY="${SAMPLE_CITY:-Buffalo}"

if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: Java 11+ is required to run Synthea." >&2
    echo "Install via: brew install openjdk@17" >&2
    exit 1
fi

if [[ ! -f "${JAR_PATH}" ]]; then
    echo "Fetching Synthea ${SYNTHEA_VERSION}..."
    curl -L -o "${JAR_PATH}" "${JAR_URL}"
fi

OUT_DIR="${REPO_ROOT}/data/sample/synthea"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "Running Synthea — population=${SAMPLE_POPULATION} seed=${SAMPLE_SEED}..."
java -jar "${JAR_PATH}" \
    --config "${SCRIPT_DIR}/synthea.properties" \
    --seed "${SAMPLE_SEED}" \
    --clinicianSeed "${SAMPLE_SEED}" \
    --population "${SAMPLE_POPULATION}" \
    --exporter.baseDirectory "${OUT_DIR}" \
    "${SAMPLE_STATE}" "${SAMPLE_CITY}"

echo "Done. CSVs at ${OUT_DIR}/csv/"
ls -lh "${OUT_DIR}/csv/" || true
