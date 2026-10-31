#!/bin/bash
# Довести все прогоны до конца и забрать результаты. Запуск повторяем: готовое пропускается, идущие ядра только ожидаются,
# обучение заново не стартует. Статус — runs/queue/status, логи прогонов — runs/queue/<прогон>.log.
#
#   systemd-run --user --unit=xlsum-queue --working-directory="$PWD" bash tools/run_queue.sh
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=runs/queue
UL_TOKEN="$HOME/.kaggle/ul_access_token"
mkdir -p "$OUT"
note() { echo "$(date '+%d.%m %H:%M') $1" >> "$OUT/status"; }

# Готов — есть журнал и прогнозы на val по всем девяти языкам.
done_run() { [ -f "runs/xlsum-$1/training_log.json" ] && [ "$(ls "runs/xlsum-$1/predictions/val" 2>/dev/null | wc -l)" -eq 9 ]; }

# step <конфиг> <аккаунт> [ключи исполнителя]: дождаться ядра, а если его ещё нет — отправить.
step() {
    local name=$1 account=$2 token=${3:-}
    shift 3 || shift 2
    if done_run "$name"; then note "$name уже забран"; return 0; fi
    local mode=()
    if KAGGLE_API_TOKEN="${token:-$HOME/.kaggle/access_token}" timeout 60 $PY -m kaggle kernels status \
        "$account/xlsum-$name" 2>&1 | grep -q 'has status'; then
        mode=(--attach)
    else
        mode=(--skip-tests)
    fi
    env -u KAGGLE_API_TOKEN $PY tools/kaggle_train.py --config "configs/experiments/$name.yaml" "${mode[@]}" "$@" \
        >> "$OUT/$name.log" 2>&1
    note "$name: код выхода $?"
}

( step mt5-small badzhik ) &
( step mt5-base uliabzh "$UL_TOKEN" --runner-user uliabzh --runner-token "$UL_TOKEN" ) &
# На badzhik не больше двух ядер сразу: дообучение идёт после эталона.
( step reference badzhik; step mt5-xlsum-finetune badzhik ) &
wait
note 'очередь завершена'
