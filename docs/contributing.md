# Contributing to Documentation

## Documentation Versioning

This project uses [mike](https://github.com/jimporter/mike) for documentation versioning. Documentation is automatically built and deployed via GitHub Actions.

### Automatic Deployment

- **Development docs**: Built on every push to `main` branch → available at `/dev/`
- **Release docs**: Built on every tag push (`v*`) → available at `/latest/` and `/v{version}/`

### Manual Version Management

If you need to manually manage documentation versions:

```bash
# Install dependencies
uv sync --extra docs

# Deploy a new version
uv run mike deploy --push --update-aliases v1.0.0 latest

# Set default version
uv run mike set-default --push latest

# List all versions
uv run mike list

# Delete a version
uv run mike delete --push v0.9.0
```

### Local Development

To build and serve documentation locally:

```bash
# Serve with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

### Version Structure

- `latest` - Latest stable release
- `dev` - Development version from main branch
- `v{X.Y.Z}` - Specific version tags

The documentation will be available at:
- https://royerlab.github.io/tracksdata/ (latest)
- https://royerlab.github.io/tracksdata/dev/ (development)
- https://royerlab.github.io/tracksdata/v1.0.0/ (specific version)
