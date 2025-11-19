"""Tests for configuration loading and management."""

import json
import tempfile
from pathlib import Path

import pytest


def test_merge_settings_basic():
    """Test basic settings merge."""
    from pump_monitor import merge_settings
    
    defaults = {"a": 1, "b": 2}
    overrides = {"b": 3, "c": 4}
    
    result = merge_settings(defaults, overrides)
    
    assert result["a"] == 1  # From defaults
    assert result["b"] == 3  # Overridden
    assert result["c"] == 4  # Added


def test_merge_settings_nested():
    """Test nested dictionary merge."""
    from pump_monitor import merge_settings
    
    defaults = {
        "mqtt": {"broker": "localhost", "port": 1883},
        "timing": {"interval": 300}
    }
    overrides = {
        "mqtt": {"broker": "192.168.1.1"},
        "timing": {"interval": 600, "new_field": 10}
    }
    
    result = merge_settings(defaults, overrides)
    
    assert result["mqtt"]["broker"] == "192.168.1.1"  # Overridden
    assert result["mqtt"]["port"] == 1883  # Preserved
    assert result["timing"]["interval"] == 600  # Overridden
    assert result["timing"]["new_field"] == 10  # Added


def test_merge_settings_none_override():
    """Test that None values in overrides are ignored."""
    from pump_monitor import merge_settings
    
    defaults = {"a": 1, "b": 2}
    overrides = {"a": None, "b": 3}
    
    result = merge_settings(defaults, overrides)
    
    assert result["a"] == 1  # None ignored
    assert result["b"] == 3  # Overridden


def test_merge_settings_empty_overrides():
    """Test merge with empty overrides."""
    from pump_monitor import merge_settings
    
    defaults = {"a": 1, "b": 2}
    overrides = {}
    
    result = merge_settings(defaults, overrides)
    
    assert result == defaults


def test_resolve_path_absolute():
    """Test path resolution with absolute path."""
    from pump_monitor import _resolve_path
    
    abs_path = Path("/tmp/test.txt")
    result = _resolve_path(abs_path)
    
    assert result.is_absolute()
    assert result == abs_path.resolve()


def test_resolve_path_relative():
    """Test path resolution with relative path."""
    from pump_monitor import _resolve_path
    
    base_dir = Path("/home/user")
    rel_path = "data/file.txt"
    
    result = _resolve_path(rel_path, base_dir)
    
    assert result.is_absolute()
    # On macOS /home might resolve to /System/Volumes/Data/home
    assert str(result).endswith("home/user/data/file.txt")


def test_resolve_path_with_tilde():
    """Test path resolution with home directory expansion."""
    from pump_monitor import _resolve_path
    
    result = _resolve_path("~/test.txt")
    
    assert result.is_absolute()
    assert "~" not in str(result)


def test_resolve_path_default_base():
    """Test path resolution without explicit base directory."""
    from pump_monitor import _resolve_path
    
    result = _resolve_path("test.txt")
    
    assert result.is_absolute()


def test_configure_from_file_valid(tmp_path):
    """Test loading valid configuration file."""
    from pump_monitor import configure_from_file
    
    config_file = tmp_path / "test_settings.json"
    config_data = {
        "mqtt": {"broker": "test.local", "port": 1884},
        "timing": {"led_check_interval_seconds": 120}
    }
    
    config_file.write_text(json.dumps(config_data))
    
    result = configure_from_file(config_file)
    
    assert result["mqtt"]["broker"] == "test.local"
    assert result["mqtt"]["port"] == 1884
    assert result["timing"]["led_check_interval_seconds"] == 120


def test_configure_from_file_missing():
    """Test loading non-existent configuration file."""
    from pump_monitor import configure_from_file
    
    with pytest.raises(FileNotFoundError):
        configure_from_file("/nonexistent/config.json")


def test_configure_from_file_invalid_json(tmp_path):
    """Test loading configuration file with invalid JSON."""
    from pump_monitor import configure_from_file
    
    config_file = tmp_path / "bad_settings.json"
    config_file.write_text("{ invalid json }")
    
    with pytest.raises(json.JSONDecodeError):
        configure_from_file(config_file)


def test_configure_from_file_invalid_structure(tmp_path):
    """Test loading configuration file with invalid structure."""
    from pump_monitor import configure_from_file
    
    config_file = tmp_path / "bad_structure.json"
    config_file.write_text(json.dumps([1, 2, 3]))  # Array instead of object
    
    # Raises TypeError from merge_settings when it tries to merge
    with pytest.raises((ValueError, TypeError), match="must be a (JSON object|dict)"):
        configure_from_file(config_file)


def test_load_state_missing_file(tmp_path, monkeypatch):
    """Test loading state when file doesn't exist."""
    from pump_monitor import load_state
    
    # Set STATE_FILE to non-existent path
    monkeypatch.setattr("pump_monitor.STATE_FILE", tmp_path / "nonexistent.json")
    
    state = load_state()
    
    assert state["last_temp_check"] is None
    assert state["last_temperature"] is None
    assert state["pump_on"] is False
    assert state["led_region"] is None


def test_load_state_valid(tmp_path, monkeypatch):
    """Test loading valid state file."""
    from pump_monitor import load_state
    
    state_file = tmp_path / "state.json"
    state_data = {
        "last_temp_check": "2023-11-15T10:30:00",
        "last_temperature": 42.5,
        "pump_on": True,
        "led_region": [100, 200, 50, 50]
    }
    state_file.write_text(json.dumps(state_data))
    
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    state = load_state()
    
    assert state["last_temp_check"] == "2023-11-15T10:30:00"
    assert state["last_temperature"] == 42.5
    assert state["pump_on"] is True
    assert state["led_region"] == [100, 200, 50, 50]


def test_load_state_missing_keys(tmp_path, monkeypatch):
    """Test loading state file with missing keys."""
    from pump_monitor import load_state
    
    state_file = tmp_path / "state.json"
    state_data = {"pump_on": True}  # Missing other keys
    state_file.write_text(json.dumps(state_data))
    
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    state = load_state()
    
    assert state["pump_on"] is True  # Preserved
    assert state["last_temp_check"] is None  # Default
    assert state["last_temperature"] is None  # Default
    assert state["led_region"] is None  # Default


def test_load_state_corrupted(tmp_path, monkeypatch):
    """Test loading corrupted state file."""
    from pump_monitor import load_state
    
    state_file = tmp_path / "state.json"
    state_file.write_text("{ corrupted json")
    
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    state = load_state()
    
    # Should return defaults
    assert state["last_temp_check"] is None
    assert state["pump_on"] is False


def test_load_state_invalid_type(tmp_path, monkeypatch):
    """Test loading state file with invalid type."""
    from pump_monitor import load_state
    
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps([1, 2, 3]))  # Array instead of object
    
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    state = load_state()
    
    # Should return defaults
    assert state["last_temp_check"] is None
    assert state["pump_on"] is False


def test_save_state_valid(tmp_path, monkeypatch):
    """Test saving valid state."""
    from pump_monitor import save_state
    
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    state = {
        "last_temp_check": "2023-11-15T10:30:00",
        "last_temperature": 42.5,
        "pump_on": True,
        "led_region": [100, 200, 50, 50]
    }
    
    save_state(state)
    
    assert state_file.exists()
    loaded = json.loads(state_file.read_text())
    assert loaded == state


def test_save_state_creates_directory(tmp_path, monkeypatch):
    """Test that save_state creates parent directories."""
    from pump_monitor import save_state
    
    state_file = tmp_path / "subdir" / "state.json"
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    state = {"pump_on": False}
    save_state(state)
    
    assert state_file.exists()


def test_save_state_invalid_type(tmp_path, monkeypatch, capsys):
    """Test saving state with invalid type."""
    from pump_monitor import save_state
    
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("pump_monitor.STATE_FILE", state_file)
    
    # Should not crash, just log error
    save_state("not a dict")
    
    captured = capsys.readouterr()
    assert "invalid state type" in captured.out.lower()


def test_save_state_none_file(monkeypatch):
    """Test saving state when STATE_FILE is None."""
    from pump_monitor import save_state
    
    monkeypatch.setattr("pump_monitor.STATE_FILE", None)
    
    # Should not crash
    save_state({"pump_on": True})
