#!/usr/bin/env bash
# Heyelan Dijital Twin — geliştirme ortamı (backend + frontend tek komut)
# Kullanım: ./run_app.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "▶ Backend başlatılıyor (http://localhost:8000) …"
python3 -m uvicorn backend.main:app --reload --port 8000 &
BACK_PID=$!

cleanup() {
  echo ""
  echo "■ Kapatılıyor …"
  kill "$BACK_PID" 2>/dev/null || true
  kill "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "▶ Frontend başlatılıyor (http://localhost:5173) …"
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "  → bağımlılıklar kuruluyor (npm install) …"
  npm install
fi
npm run dev &
FRONT_PID=$!

echo ""
echo "──────────────────────────────────────────────"
echo "  Frontend : http://localhost:5173"
echo "  Backend  : http://localhost:8000/api/health"
echo "  Kapatmak için: Ctrl+C"
echo "──────────────────────────────────────────────"

wait
