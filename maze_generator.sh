#!/bin/bash

# --- Configuration ---
NB_GRIDS=100000         # nombre de labyrinthes à générer
NPROC=8               # nombre de processus en parallèle
PROJECT_DIR="/home/louis/Documents/Programs/365/MAZE"
LOG_DIR="$PROJECT_DIR/logs"

OUT_LOG="$LOG_DIR/output.log"
ERR_LOG="$LOG_DIR/error.log"
TIME_LOG="$LOG_DIR/time.log"

# =========================
# Préparation
# =========================
mkdir -p "$LOG_DIR"

# Nettoyage anciens logs
> "$OUT_LOG"
> "$ERR_LOG"
> "$TIME_LOG"

# --- Environnement ---
cd "$PROJECT_DIR" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MAZE

# =========================
# Chrono début
# =========================
START_TIME=$(date +%s)

echo "Starting maze generation at $(date)" >> "$OUT_LOG"
echo "Using $NPROC parallel processes" >> "$OUT_LOG"
echo "Generating $NB_GRIDS grids" >> "$OUT_LOG"
echo "---------------------------------" >> "$OUT_LOG"

# =========================
# Génération parallèle
# =========================
seq 1 "$NB_GRIDS" | xargs -P "$NPROC" -I {} \
bash -c 'python maze_generator.py {} >> "'"$OUT_LOG"'" 2>> "'"$ERR_LOG"'"'

# =========================
# Chrono fin
# =========================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "---------------------------------" >> "$OUT_LOG"
echo "Finished at $(date)" >> "$OUT_LOG"
echo "Total time: ${ELAPSED} seconds" >> "$OUT_LOG"

echo "Total time (seconds): $ELAPSED" > "$TIME_LOG"