PROJECT_NAME = pymetal
VERSION = 0.1.0

.PHONY: all build wheel clean test snap

all: build


build:
	@uv sync --reinstall-package pymetal

wheel:
	@uv build

install:
	@uv pip uninstall pymetal || true
	@uv pip install dist/pymetal-*.whl

clean:
	rm -rf build dist src/pymetal/*.so

test:
	@.venv/bin/pytest tests/ -v

repl:
	@uv run python -m pychuck tui

snap:
	@git add --all . && git commit -m 'snap' && git push
