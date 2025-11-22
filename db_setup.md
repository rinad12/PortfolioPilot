# Quick Run Guide

Use this cheat sheet for daily development operations once your environment is configured.

## 1. Start the Database

Run this to start the database in the background.

```bash
docker-compose up -d db
```

## 2. Check Logs

If something seems wrong, tail the logs in real-time. (Press Ctrl+C to exit).

```bash
docker logs -f app_db_local
```

## 3. Stop the Database

Stops the container but keeps your data safe.

```bash
docker-compose stop
```

## 4. Connect via Command Line

Access the running database directly from your terminal.

```bash
docker exec -it app_db_local psql -U admin -d app_db
```

## 5. Reset / Wipe Data (Danger Zone)

Run this only if you want to delete the database and start fresh (e.g., if you messed up a migration).

```bash
docker-compose down -v
docker-compose up -d db
```

## 6. Connection Strings

| Service | URL / Value |
|---------|-------------|
| Local Host | `postgresql://admin:secret@localhost:5432/app_db` |
| Inside Docker | `postgresql://admin:secret@db:5432/app_db` |


