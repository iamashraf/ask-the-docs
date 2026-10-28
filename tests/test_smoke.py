from ask_the_docs import __name__ as package_name


def test_package_imports() -> None:
    assert package_name == "ask_the_docs"
