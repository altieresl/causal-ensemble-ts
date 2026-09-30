# Frontend (React + TypeScript + Vite)

SPA que consome a API em `backend/` (`/api/v1`). Arquitetura: `.local/ARQUITETURA_WEB.md`.

```powershell
npm install
npm run dev        # http://localhost:5173 (proxy /api -> http://localhost:8000; VITE_API_PROXY muda o alvo)
npm run typecheck && npm test
npm run build      # gera dist/, servido pelo FastAPI quando existe
```

Estrutura: `src/api` (cliente tipado, tipos espelhando os schemas do backend e hooks do
TanStack Query, com polling do status até estado terminal), `src/features/{datasets,runs,results}`,
`src/components` (UI genérica) e `src/lib` (funções puras, testadas: layout e colapso de arestas
do grafo, formatação).
