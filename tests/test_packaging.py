def test_package_is_importable() -> None:
    """The editable install produces a package you can actually import.

    Not a placeholder: this fails if the hatch `packages` path is wrong, if
    __init__.py is missing, or if the src layout isn't wired up — all things
    `pip install -e` will happily report as success.
    """
    import ratchet

    assert ratchet is not None

def test_the_dev_extra_installs_what_the_source_imports() -> None:
    """CI installs only `[dev]`, so anything the source imports must arrive with it.

    This caught a green local run failing in CI: `dev` restated the agent
    dependency list instead of referencing it, the agent extra moved from `openai`
    to `langchain`, and `dev` kept the stale line. Type-checking then ran without
    the libraries the source imports. Two lists that must stay identical will not.
    """
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev = config["project"]["optional-dependencies"]["dev"]

    assert any(d.replace(" ", "").startswith("ratchet[agent]") for d in dev), (
        "dev must reference the agent extra rather than duplicating its contents"
    )
