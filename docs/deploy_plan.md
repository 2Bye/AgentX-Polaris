# 📦 Deploy Plan — Purple Agent (AgentBeats τ²-Bench)

Документ отслеживает прогресс подготовки агента к публикации на соревновательной платформе.

## Чеклист

- [x] **Шаг 0:** Локальное тестирование — Pass Rate 100% (2/2), модель `gpt-5.4`
- [x] **Шаг 1:** Фикс GitHub Actions CI (`test-and-publish.yml`)
- [x] **Шаг 2:** Полноценный `amber-manifest.json5`
- [ ] **Шаг 3:** Инициализация Git-репозитория и push на GitHub
- [ ] **Шаг 4:** Верификация Docker-образа локально
- [ ] **Шаг 5:** Регистрация на `agentbeats.dev`

---

## Шаг 1 — Фикс GitHub Actions CI

**Проблема:** В `test-and-publish.yml` проверка agent-card идёт по неверному пути `/agent.json`, тогда как A2A SDK отдаёт её по `/.well-known/agent-card.json`. Также порт у агента `9019` (в `server.py`) расходится с `9009`, прописанным в CI.

**Решение:** Исправить путь curl на `/.well-known/agent-card.json` и согласовать порт на `9009` (Dockerfile уже использует этот порт).

### Изменения в `.github/workflows/test-and-publish.yml`

| Было | Стало |
|---|---|
| `--port 9009 &` | `--port 9009 &` (без изменений) |
| `curl -f http://127.0.0.1:9009/.well-known/agent.json` | `curl -f http://127.0.0.1:9009/.well-known/agent-card.json` |

**Статус:** ✅ Выполнено

---

## Шаг 2 — Полноценный `amber-manifest.json5`

**Проблема:** Текущий `amber-manifest.json5` — заглушка без обязательных полей `manifest_version`, `config_schema`, `program`, `provides`, `exports`.

**Решение:** Переписать по образцу из `_template/amber-manifest.json5`, заполнив Docker-образ, порт, секрет API-ключа, entrypoint.

**Статус:** ✅ Выполнено

### Ключевые поля манифеста

| Поле | Значение |
|---|---|
| `manifest_version` | `"0.1.0"` |
| `image` | `ghcr.io/<GITHUB_USERNAME>/<REPO_NAME>:latest` |
| `entrypoint` | `uv run src/server.py --host 0.0.0.0 --port 9009` |
| `port` | `9009` |
| `secret` | `OPENAI_API_KEY` |

> ⚠️ **Требует действия пользователя:** Заменить `<GITHUB_USERNAME>/<REPO_NAME>` на реальные значения после создания репозитория.

---

## Шаг 3 — Git и GitHub (предстоит)

Необходимо:
1. `git init && git add . && git commit -m "feat: Purple Agent v1.0.0"`
2. Создать репозиторий на GitHub (Settings → Packages должны быть включены)
3. `git remote add origin <URL>` и `git push -u origin main`
4. Убедиться, что GitHub Actions запустился и образ опубликован в `ghcr.io`

---

## Шаг 4 — Docker (предстоит)

```bash
docker build --platform linux/amd64 -t purple-agent .
docker run --env-file .env -p 9009:9009 purple-agent
curl http://localhost:9009/.well-known/agent-card.json
```

---

## Шаг 5 — Регистрация (предстоит)

После публикации образа — зарегистрировать агента на `agentbeats.dev` с готовым `amber-manifest.json5`.
