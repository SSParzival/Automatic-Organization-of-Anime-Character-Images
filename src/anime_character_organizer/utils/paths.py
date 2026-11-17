"""
Filesystem path normalization and portability helpers.
"""

from pathlib import Path
from typing import Optional, Union


def normalize_path_string(value: Union[str, Path]) -> str:
    """Resolve and expand a path, returning a normalized POSIX string."""
    return checked_path(str(value)).as_posix()


def path_to_posix(path: Union[str, Path]) -> str:
    """Convert a path to POSIX string representation."""
    return Path(path).as_posix()


def safe_relative_path(path: Union[str, Path], base_dir: Union[str, Path]) -> str:
    """
    Compute a relative path from base_dir if possible.
    If path is not under base_dir, return normalized POSIX path.
    """
    path = checked_path(path)
    base_dir = checked_path(base_dir)
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def checked_path(value: Union[str, Path], *, allow_leaf_symlink: bool = False) -> Path:
    """Return an absolute path without following symlinks or accessing sandbox.

    Symlink ancestors are rejected, including aliases to protected directories.
    A leaf symlink is allowed only for explicit unlink/replace operations.
    """
    import os

    raw = Path(value).expanduser().absolute()
    if "sandbox" in raw.parts:
        raise ValueError("Paths containing a sandbox directory are protected.")
    path = Path(os.path.abspath(raw))
    for ancestor in reversed(path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symlink directory is not supported: {ancestor}")
    if path.is_symlink() and not allow_leaf_symlink:
        raise ValueError(f"Symlink path is not supported: {path}")
    return path


def notebook_paths() -> tuple[Path, Path]:
    """Resolve explicit notebook paths independently of Jupyter's working directory.

    ANIME_PIPELINE_PROJECT_DIR is required and absolute. INPUT_DIR defaults to
    input_images beside that workspace; a supplied override must also be absolute.
    """
    import os

    from ..exceptions import ConfigurationError

    workspace = os.getenv("ANIME_PIPELINE_PROJECT_DIR")
    if not workspace:
        raise ConfigurationError(
            "Set ANIME_PIPELINE_PROJECT_DIR to an absolute workspace path before starting Jupyter."
        )
    project = Path(workspace).expanduser()
    source = Path(os.getenv("ANIME_PIPELINE_INPUT_DIR", str(project.parent / "input_images"))).expanduser()
    if not project.is_absolute() or not source.is_absolute():
        raise ConfigurationError("Notebook input and workspace paths must be absolute.")
    return checked_path(source), checked_path(project)


def ensure_empty_output_dir(path: Union[str, Path]) -> Path:
    """Validate that an output directory is safe and either new or empty."""
    from ..exceptions import ConfigurationError

    checked = checked_path(path)
    if checked.exists():
        if not checked.is_dir():
            raise ConfigurationError(f"Output path exists and is not a directory: {checked}")
        if any(checked.iterdir()):
            raise ConfigurationError("Output directory must be new or empty to protect existing artifacts.")
    return checked


def load_dotenv(env_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Load KEY=VALUE pairs from a .env file into os.environ without overwriting existing vars."""
    import os

    from ..exceptions import ConfigurationError

    if env_path is not None:
        candidate = checked_path(env_path)
        if not candidate.is_file():
            raise ConfigurationError(f"Environment file not found: {candidate}")
    else:
        cwd_env = Path.cwd() / ".env"
        repo_env = Path(__file__).resolve().parents[3] / ".env"
        if cwd_env.is_file():
            candidate = cwd_env
        elif repo_env.is_file():
            candidate = repo_env
        else:
            return None
        candidate = checked_path(candidate)

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)

    return candidate
