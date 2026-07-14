"""Keep the public README workflow synchronized with the package."""

from examples.readme_quickstart import run_quickstart


def test_readme_quickstart_executes_end_to_end():
    result = run_quickstart()

    assert result["method"] == "M1"
    assert result["n_site_bags"] == 25
    assert result["n_background_bags"] == 25
    assert result["prediction_shape"] == [24, 24]
    assert 0.0 <= result["minimum_relative_suitability"] <= 1.0
    assert 0.0 <= result["maximum_relative_suitability"] <= 1.0
    assert result["minimum_relative_suitability"] < result["maximum_relative_suitability"]
