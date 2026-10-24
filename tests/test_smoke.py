from research_assistant import __name__ as package_name


def test_package_imports() -> None:
    assert package_name == "research_assistant"
