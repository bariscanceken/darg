#!/usr/bin/env bash
# Update DARG on the server: pull the latest code from GitHub, rebuild, restart. Run from anywhere:
#   ./deploy/guncelle.sh
# Data (documents, index, database) is not touched.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Kod GitHub'dan çekiliyor"
git pull --ff-only

echo "==> Kutu (image) yeniden kuruluyor ve sistem yeniden başlatılıyor"
docker compose up -d --build

echo "==> Eski kutular temizleniyor"
docker image prune -f >/dev/null

echo "==> Sistemin açılması bekleniyor (modeller yükleniyor)"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/saglik >/dev/null 2>&1; then
    echo "==> DARG çalışıyor: $(git log -1 --format='%h %s')"
    docker compose ps
    exit 0
  fi
  sleep 2
done
echo "!! DARG 2 dakikada açılmadı. Kayıtlara bakın: docker compose logs --tail 50 api" >&2
exit 1
