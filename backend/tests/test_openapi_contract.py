from backend.scripts.export_openapi import TARGET, build_spec


def test_committed_openapi_matches_current_contract():
    """Se falhar: rode `npm run generate:api` em frontend/ e commite openapi.json + schema.d.ts."""
    assert TARGET.read_text(encoding="utf-8").replace("\r\n", "\n") == build_spec()
