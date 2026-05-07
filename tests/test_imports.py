def test_package_imports():
    import sentiment
    assert sentiment.__version__


def test_modules_importable():
    from sentiment import (  # noqa: F401
        config,
        data,
        embeddings,
        models,
        predict,
        preprocessing,
        train,
    )
