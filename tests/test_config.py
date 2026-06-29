from src.config import load_config, get_project_path


def test_config_loads():
    config = load_config()

    assert "paths" in config
    assert "model" in config
    assert "api" in config
    assert config["api"]["app_name"] == "life-expectancy-api"


def test_project_path_resolves():
    path = get_project_path("config/config.yaml")

    assert path.exists()
