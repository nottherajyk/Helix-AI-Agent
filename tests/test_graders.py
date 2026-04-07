from helixdesk.graders import run_all_graders, check_api_compliance

def test_check_api_compliance():
    res = check_api_compliance()
    assert all(r.passed for r in res)

def test_all_graders():
    res = run_all_graders(2)
    assert "total_score" in res
