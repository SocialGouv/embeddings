# Chroma Embeddings

## Installation

### Environment variables
Create a `./.env` file containing the following variables
```bash
DATA_DIRECTORY=./my_md_files # folder storing files md to index
CHROMA_PERSIST_DIRECTORY=./.database # folder storing indexes
```

### With docker

```shell
docker compose --build -d
```

### Manual
```shell
python -m venv <my_env_name>
source <my_env_name>/bin/activate
pip install -r requirements.txt
python app.py
```

## API

| Method | Route                        | Description                         |
|--------|------------------------------|-------------------------------------|
| GET    | /healthz                     | Healthz check                       |
| GET    | /api/collections             | List all collections in database    |
| GET    | /api/collection/:name?query= | Run `query` over `:name` collection |
| GET    | /api/collection/:name/info   | Get info about `:name` collection   |
| GET    | /api/collection/:name/index  | Create indexes from `DATA_DIRECTORY` and add indexes to `:name` collection |
