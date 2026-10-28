"""Registration must be explicit and preserve state when validation fails."""
import pytest
from tlg_api.algorithms import _registry


REGISTRATIONS = [
    ("register_dynamic_system", "_DYNAMIC_SYSTEMS"),
    ("register_extractor", "_EXTRACTORS"),
    ("register_modeler", "_MODELERS"),
    ("register_merger", "_MERGERS"),
]


@pytest.mark.parametrize("function_name,registry_name", REGISTRATIONS)
def test_duplicate_registration_requires_explicit_replacement(monkeypatch, function_name, registry_name):
    registry = {}
    monkeypatch.setattr(_registry, registry_name, registry)
    monkeypatch.setattr(_registry, "_SCORERS", {})
    register = getattr(_registry, function_name)
    original = lambda: "original"
    replacement = lambda: "replacement"
    assert register("sample", original) is original
    with pytest.raises(ValueError, match="already registered"):
        register("sample", replacement)
    assert registry["sample"] is original
    assert register("sample", replacement, replace=True) is replacement
    assert registry["sample"] is replacement


@pytest.mark.parametrize("function_name,registry_name", REGISTRATIONS)
def test_invalid_registration_does_not_modify_registry(monkeypatch, function_name, registry_name):
    original = lambda: None
    registry = {"sample": original}
    monkeypatch.setattr(_registry, registry_name, registry)
    monkeypatch.setattr(_registry, "_SCORERS", {})
    register = getattr(_registry, function_name)
    for bad_name in ("", "   ", None, 42):
        with pytest.raises(ValueError, match="non-empty string"):
            register(bad_name, original, replace=True)
    with pytest.raises(TypeError, match="callable"):
        register("new", object())
    with pytest.raises(TypeError, match="callable"):
        register("sample", object(), replace=True)
    assert registry == {"sample": original}


def test_modeler_replacement_clears_scorer_and_rejects_invalid_scorer_atomically(monkeypatch):
    models, scorers = {}, {}
    monkeypatch.setattr(_registry, "_MODELERS", models)
    monkeypatch.setattr(_registry, "_SCORERS", scorers)
    original = lambda: None
    replacement = lambda: None
    scorer = lambda values: values
    _registry.register_modeler("sample", original, scorer=scorer)
    with pytest.raises(TypeError, match="scorer must be callable"):
        _registry.register_modeler("sample", replacement, scorer=object(), replace=True)
    assert models == {"sample": original}
    assert scorers == {"sample": scorer}
    with pytest.raises(TypeError, match="scorer must be callable"):
        _registry.register_modeler("new", replacement, scorer=object())
    assert "new" not in models and "new" not in scorers
    _registry.register_modeler("sample", replacement, replace=True)
    assert models["sample"] is replacement
    assert _registry.get_scorer("sample") is None
