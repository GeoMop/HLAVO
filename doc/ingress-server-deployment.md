# Ingress Server

## Overview

Ingress Server is a Flask application for uploading CSV/JSON data into an S3-backed Zarr store. It serves as a backend service for collecting and processing scientific data.

## Deployment

The application is deployed on e-infra Rancher with the following configuration:

- **Namespace**: `hlavo-ingress-server`
- **URL**: https://zarr-fuse-hlavo-ingress-server.dyn.cloud.e-infra.cz/
- **Source Code**: https://github.com/GeoMop/zarr_fuse (branch `main`, path `app/databuk/ingress_server`)

## Application Access

The application is available at:
```
https://zarr-fuse-hlavo-ingress-server.dyn.cloud.e-infra.cz/
```

## Helm Chart

The Helm chart for deployment is available in the zarr_fuse repository:

- **Repository**: https://github.com/GeoMop/zarr_fuse
- **Chart Location**: `app/databuk/ingress_server/charts/ingress-server`
- **Chart Documentation**: See `values.yaml` for detailed configuration options

The chart handles deployment configuration including service, ingress (NGINX with cert-manager for Let's Encrypt SSL), persistence, and security context.

## Deployment Workflow

The application is deployed via GitHub Actions workflow: `.github/workflows/ingress-server-push-main.yaml`

This workflow uses the reusable workflow from zarr_fuse repository:
- **Reusable Workflow**: `GeoMop/zarr_fuse/.github/workflows/ingress-server-reusable-workflow.yaml`

### Workflow Configuration

The workflow uses `secrets: inherit` to pass secrets to the reusable workflow. The reusable workflow handles:
- Building and pushing the Docker image
- Deploying to e-infra Rancher using Helm
- Managing S3 credentials and configuration passed from secrets

### Required Secrets

The following secrets must be configured in the GitHub repository settings (Settings → Secrets and variables → Actions):

| Secret                  | Description                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `DOCKER_USERNAME`       | Docker Hub username for pushing images                                                                 |
| `DOCKER_PASSWORD`       | Docker Hub password or token for pushing images                                                        |
| `S3_ACCESS_KEY`         | S3 access key for accessing the Zarr store                                                             |
| `S3_SECRET_KEY`         | S3 secret key for accessing the Zarr store                                                             |
| `KUBECONFIG`            | Base64-encoded kubeconfig for e-infra Rancher cluster access                                           |
| `BASIC_AUTH_USERS_JSON` | JSON string of users for basic authentication (format: `{"user1": "password1", "user2": "password2"}`) |

### Required Variables

The following variables must be configured in the GitHub repository settings (Settings → Secrets and variables → Actions):

| Variable          | Description                                | Example                       |
| ----------------- | ------------------------------------------ | ----------------------------- |
| `DOCKER_USERNAME` | Docker Hub username (can also be a secret) | `jbrezmorf`                   |
| `S3_ENDPOINT_URL` | S3 endpoint URL                            | `https://s3.cl4.du.cesnet.cz` |

### Workflow Inputs

The reusable workflow accepts the following inputs (configured in `.github/workflows/ingress-server-push-main.yaml`):

| Input                    | Description                                            | Current Value                              |
| ------------------------ | ------------------------------------------------------ | ------------------------------------------ |
| `deploy`                 | Whether to deploy the Helm chart                       | `true`                                     |
| `tag`                    | Container image tag (use `generate` for CI-based tags) | `generate`                                 |
| `namespace`              | Kubernetes namespace                                   | `hlavo-ingress-server`                     |
| `release-name`           | Helm release name                                      | `hlavo-ingress-server`                     |
| `s3-store-url`           | S3 store URL (Zarr store location)                     | `s3://app-databuk-test-service/hlavo.zarr` |
| `docker-repository`      | Docker repository for the image                        | `jbrezmorf/hlavo-ingress-server`           |
| `docker-registry`        | Docker registry                                        | `docker.io`                                |
| `configuration-dir-path` | Path to configuration files                            | `data_processing/ingress/inputs`           |
| `zarr-fuse-ref`          | zarr_fuse repository branch/tag                        | `SM-CI-outside-repository`                 |
| `extra-helm-args`        | Additional Helm arguments                              | (sets image.name)                          |

## Configuration Files

Configuration files are located in `data_processing/ingress/inputs/` and are bundled into the Docker image during the workflow execution.

### Configuration Directory Structure

```
data_processing/ingress/inputs/
├── endpoints_config.yaml       # Endpoint and scraper configuration
├── requirements.txt            # Python dependencies for extractors
├── schemas/                    # Data schemas
│   └── hlavo_surface_schema.yaml
└── dataframes/                 # Sample dataframes
    └── hlavo_surface_dataframe.csv
```

### Key Configuration Files

#### `endpoints_config.yaml`
Defines active data scrapers and endpoints:
- Scraper names and scheduling (cron expressions)
- API endpoints to fetch data from
- Data extraction functions and modules
- Schema definitions
- Location dataframes (CSV files with measurement point coordinates)
- Example: Fetches weather forecast data from yr.no API at scheduled times

**Current Configuration:**
- Active scraper: `hlavo-surface-forecast` (weather data)
- Schedule: Multiple daily updates via cron jobs
- Data source: yr.no weather API
- Schema: `hlavo_surface_schema.yaml`

#### `schemas/hlavo_surface_schema.yaml`
Defines the Zarr storage structure and variables:
- Coordinate systems and grids
- Variable definitions (temperature, precipitation, pressure, cloud fraction, etc.)
- Units and data types
- S3 endpoint URL for storage

**Current Schema:**
- S3 endpoint: CESNET S3 (https://s3.cl4.du.cesnet.cz)
- Grid coverage: Approx. 50.84°N-50.89°N, 14.85°E-14.96°E (approximately 1km resolution)
- Variables: Weather parameters from yr.no (temperature, precipitation, pressure, clouds)

#### `dataframes/hlavo_surface_dataframe.csv`
CSV file containing measurement point metadata:
- Measurement site identifiers and coordinates
- Environmental characteristics (forest, meadow, field, etc.)
- Terrain properties (flat, slope, dip)
- Soil properties
- Comments and notes

**Current Data:**
- Contains HLAVO project measurement points with GPS coordinates
- Used as reference data for spatial indexing in the Zarr store

## More Information

- **Source Code**: https://github.com/GeoMop/zarr_fuse
- **Helm Chart**: https://github.com/GeoMop/zarr_fuse/tree/main/app/databuk/ingress_server/charts
- **Ingress Server Documentation**: https://github.com/GeoMop/zarr_fuse/tree/main/app/databuk/ingress_server
