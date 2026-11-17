//! Regression test for finding 14: `e2e` refuses an `arrivalgen` from another build.
use sidecar_streaming::e2e::check_arrivalgen;
use std::os::unix::fs::PermissionsExt;

fn script(name: &str, line: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fix_e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/sh\necho '{line}'\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

#[test]
fn arrivalgen_is_version_checked() {
    let same = script("same", &arrival_model::schedule::version_line());
    assert!(check_arrivalgen(&same).is_ok());
    let old = script("old", "arrival_model 0.0.1 schedule-format 0");
    let err = check_arrivalgen(&old).unwrap_err();
    assert!(err.contains("different build"), "{err}");
    assert!(check_arrivalgen(std::path::Path::new("/nonexistent/arrivalgen")).is_err());
}
