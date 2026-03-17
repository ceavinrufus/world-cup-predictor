from wcp.data.teams import canonical, canonical_many


def test_passthrough_when_already_canonical():
    assert canonical("United States") == "United States"
    assert canonical("Brazil") == "Brazil"


def test_common_aliases():
    assert canonical("USA") == "United States"
    assert canonical("Korea Republic") == "South Korea"
    assert canonical("Cote d'Ivoire") == "Ivory Coast"
    assert canonical("Czechia") == "Czech Republic"
    assert canonical("Turkey") == "Türkiye"
    assert canonical("Ireland") == "Republic of Ireland"


def test_case_insensitive_and_whitespace():
    assert canonical(" usa ") == "United States"
    assert canonical("CZECHIA") == "Czech Republic"


def test_unknown_teams_pass_through():
    assert canonical("Atlantis") == "Atlantis"


def test_batch():
    out = canonical_many(["USA", "Brazil", "Czechia"])
    assert out == ["United States", "Brazil", "Czech Republic"]
