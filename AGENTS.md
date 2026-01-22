Knowledge:
- The project uses pixi for development environment management, specifically to handle mixed Python and R dependencies.
- if pixi does not exists (check `which pixi`), set it up with: `curl -fsSL https://pixi.sh/install.sh | sh`
- To run the project's tests, use the command `pixi run test`.
- To run tests with coverage, use the command `pixi run test-cov`.
- Documentation is built using the command `pixi run -e docs docs`.
- Project versioning is configured dynamically using setuptools-git-versioning.
- The project uses pyproject.toml exclusively for configuration and dependency management,