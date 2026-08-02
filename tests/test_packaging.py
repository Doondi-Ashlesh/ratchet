def test_package_is_importable() -> None:
    """The editable install produces a package you can actually import.

    Not a placeholder: this fails if the hatch `packages` path is wrong, if
    __init__.py is missing, or if the src layout isn't wired up — all things
    `pip install -e` will happily report as success.
    """
    import ratchet

    assert ratchet is not None