# HLAVO configuration of the dashboard instance

## Local deployment

1. rename `.env.example` to `.env` put there secrets as those in `.secrets_env`
2. use `setup_env.sh` to create environment `venv`; that installs dashboard from the zarr_fuse repo
3. start dashboard server: `venv/bin/zf-dashboard`
4. open in browser: http://localhost:5006

## Dashboard config

Dashboard config for individual schemas is in `config/endpoints.yaml`,
allowing to configure coords of individual node datasets to logical dashboard axes.

