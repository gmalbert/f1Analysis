def test_service_status_refresh_auth_and_metrics(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from app.enhancements import service
    root = tmp_path
    dataset = root/"data_files"
    models = dataset/"models"
    models.mkdir(parents=True)
    (dataset/"f1ForAnalysis.csv").write_text("year\n2026\n")
    (root/"raceAnalysis.py").write_text("# source")
    (models/"manifest.json").write_text(json.dumps({"model_name":"position","notes":["recorded"],"trained_at":"today"}))
    monkeypatch.setattr(service,"DATA_DIR",dataset)
    monkeypatch.setattr(service,"REPO_ROOT",root)
    monkeypatch.delenv("F1_ADMIN_TOKEN",raising=False)
    enhancement = service.Enhancements(poll_seconds=0)
    app = FastAPI()
    enhancement.install(app)
    try:
        with TestClient(app) as client:
            first = client.get("/api/enhancements/status").json()
            assert first["dataset"]["name"] == "f1ForAnalysis.csv"
            (dataset/"f1ForAnalysis.csv").write_text("year\n2025\n2026\n")
            assert client.get("/api/enhancements/status").json()["revision"] != first["revision"]
            assert client.get("/api/enhancements/metrics").status_code == 503
            monkeypatch.setenv("F1_ADMIN_TOKEN","test-only")
            assert client.get("/api/enhancements/metrics").status_code == 403
            assert client.get("/api/enhancements/metrics",headers={"X-F1-Admin-Token":"test-only"}).status_code == 200
            assert client.post("/api/enhancements/jobs",json={"task":"unsupported"},headers={"X-F1-Admin-Token":"test-only"}).status_code == 400
            assert client.post("/api/enhancements/jobs",json={"task":"leakage-audit","values":{"Uploaded CSV":"year\n2026"}},headers={"X-F1-Admin-Token":"test-only"}).status_code == 400
            assert client.get("/api/enhancements/jobs/missing",headers={"X-F1-Admin-Token":"test-only"}).status_code == 404
    finally:
        enhancement.jobs.close()


def test_research_dispatch_uses_only_existing_opt_in_actions(monkeypatch):
    from app.enhancements import service
    monkeypatch.setattr(service,"artifact_revision",lambda *_:"revision")
    monkeypatch.setattr(service,"clear_source_caches",lambda:None)
    monkeypatch.setattr(service.presentation,"render_view",lambda page,values,action:{"page":page,"values":values,"action":action})
    context = {"revision":"revision","values":{}}
    bins = service.execute_research("bin-comparison",context)
    assert bins["action"] == "Run Bin Count Comparison"
    assert bins["values"]["Select q values (number of bins)"] == [2]
    assert service.execute_research("leakage-audit",context)["action"] == "Run Leakage Audit"
    with pytest.raises(ValueError, match="Unsupported research task"):
        service.execute_research("unknown",context)
    with pytest.raises(ValueError, match="q values"):
        service.execute_research("bin-comparison",{"revision":"revision","values":{"Select q values (number of bins)":[1]}})
    with pytest.raises(ValueError, match="Audit row limit"):
        service.execute_research("leakage-audit",{"revision":"revision","values":{"Rows to read (0 = all)":-1}})
    with pytest.raises(ValueError, match="Artifacts changed"):
        service.execute_research("leakage-audit",{"revision":"stale","values":{}})
