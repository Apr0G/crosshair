"""
P10.1.6: failures exit non-zero with one redacted line and no traceback in normal mode.

    .venv\\Scripts\\python.exe test\\engine\\test_cli_errors.py

Missing file -> 2; bad tier / unknown option / unknown command -> 2; an unfinished
delegate (missing module, no main, NotImplementedError) -> 2; an unexpected exception
-> 1 with the error on one line (traceback only with --debug, and in the log file).
"""
import atexit
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from _harness import ROOT, check, section, summary

TMP = Path(tempfile.mkdtemp(prefix="crosshair_cli_err_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)       # LOW-77
ENV = {**os.environ, "CROSSHAIR_ENGINE_DB": str(TMP / "engine.db")}
ENV.pop("CROSSHAIR_STUB_PREDICTOR", None)
os.environ["CROSSHAIR_ENGINE_DB"] = ENV["CROSSHAIR_ENGINE_DB"]

from engine import cli    # noqa: E402

LOGS = ["--log-dir", str(TMP / "logs")]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "engine.cli", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=ENV)


def no_tb(p: subprocess.CompletedProcess) -> bool:
    return "Traceback" not in p.stderr and "Traceback" not in p.stdout


section("missing files")
for args in (("analyze", str(TMP / "missing.dem"), "--out", str(TMP / "o_missing")), ("parse", str(TMP / "missing_dir")),
             ("ingest", str(TMP / "missing.dem"))):
    p = run(*args, *LOGS)
    check(f"{args[0]} on a missing path exits 2", p.returncode == 2, p.stderr[-200:])
    check(f"{args[0]}: no traceback", no_tb(p))
    check(f"{args[0]}: one ERROR line naming the file", p.stderr.count("[ERROR") == 1 and "missing" in p.stderr)
    check(f"{args[0]}: no absolute temp path in the error", str(TMP) not in p.stderr.split("[ERROR", 1)[-1].splitlines()[0])

section("usage errors")
for args, what in ((("analyze", "x.dem", "--tier", "bogus", "--out", str(TMP / "o_tier")), "bad tier"),
                   (("ingest", "x.dem", "--tier", "faceit_8_9"), "bad ingest tier"),
                   (("status", "--bogus"), "unknown option"),
                   (("frobnicate",), "unknown command")):
    p = run(*args)
    check(f"{what} exits 2", p.returncode == 2, p.stderr[-200:])
    check(f"{what}: no traceback", no_tb(p))

section("analyze without a model")
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--out", str(TMP / "o"), *LOGS)
check("no model -> exit 2", p.returncode == 2)
check("message says how to pass a model", "--model" in p.stderr and no_tb(p))
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--model-dir", str(TMP / "no_such_model"),
        "--out", str(TMP / "o_nomodel"), *LOGS)                  # --out on every analyze (LOW-62)
check("missing model directory -> exit 2", p.returncode == 2 and "model directory not found" in p.stderr)
(TMP / "wp_v99").mkdir()
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--model-dir", str(TMP / "wp_v99"),
        "--out", str(TMP / "o_unadopted"), *LOGS)
check("unadopted model -> exit 2", p.returncode == 2 and "not adopted" in p.stderr, p.stderr[-300:])
p = run("analyze", "syn.dem", "--raw-tables-story", "base", "--q-model", "q_v1", "--out", str(TMP / "o_q"), *LOGS)
check("--q-model without --neighbours -> exit 2", p.returncode == 2 and "--neighbours" in p.stderr)

section("test hooks never touch production locations (MED-03)")
HOOK_ENV = {**ENV, "CROSSHAIR_STUB_PREDICTOR": "1"}
p = subprocess.run([sys.executable, "-m", "engine.cli", "analyze", "syn.dem", "--raw-tables-story", "base", *LOGS],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=HOOK_ENV)
check("hook without --out -> exit 2, says --out", p.returncode == 2 and "--out" in p.stderr and no_tb(p), p.stderr[-300:])
NO_DB_ENV = {k: v for k, v in HOOK_ENV.items() if k != "CROSSHAIR_ENGINE_DB"}
p = subprocess.run([sys.executable, "-m", "engine.cli", "analyze", "syn.dem", "--raw-tables-story", "base",
                    "--out", str(TMP / "hook_out"), *LOGS],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=NO_DB_ENV)
rep = json.loads((TMP / "hook_out" / "report.json").read_text(encoding="utf-8")) if (TMP / "hook_out" / "report.json").is_file() else {}
check("hook with the production DB: runs, warns, writes no action_scores",
      p.returncode == 0 and "NOT written" in p.stderr and "db_rows=0" in p.stderr, p.stderr[-300:])
check("report carries the test_hook warning", any(w["code"] == "test_hook" for w in rep.get("warnings", [])))

section("configuration errors exit 2 (MED-05)")
broken = TMP / "broken.yaml"
broken.write_text("sampling:\n  hz: [1, 2\n", encoding="utf-8")
p = run("status", "--config", str(broken), *LOGS)
check("YAML syntax error -> exit 2 with line and column", p.returncode == 2 and "YAML syntax error" in p.stderr
      and "line" in p.stderr and no_tb(p), p.stderr[-300:])
NOKEY_ENV = {**ENV, "FACEIT_API_KEY": ""}
p = subprocess.run([sys.executable, "-m", "engine.cli", "scrape", "--source", "api", "--max", "1", *LOGS],
                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=NOKEY_ENV)
check("scrape without FACEIT_API_KEY -> exit 2 before any request", p.returncode == 2 and "FACEIT_API_KEY" in p.stderr
      and no_tb(p), p.stderr[-300:])

section("unfinished delegates")
p = run("eval", "--attribution", *LOGS)
check("eval --attribution (no module yet) exits 2", p.returncode == 2, p.stderr[-200:])
check("says the module does not exist", "does not exist yet" in p.stderr and no_tb(p))
p = run("eval", *LOGS)
check("eval without a target exits 2", p.returncode == 2)
real_root = cli.paths.ROOT
cli.paths.ROOT = TMP                                  # no bench/ scripts here
err = io.StringIO()
with contextlib.redirect_stderr(err):
    rc = cli.main(["bench", "value", *LOGS])
cli.paths.ROOT = real_root
check("bench with no script exits 2", rc == 2 and "does not exist" in err.getvalue(), err.getvalue()[-200:])


class _Lg:
    def info(self, *a, **k): pass


for name, mod in (("missing", None), ("no_main", types.ModuleType("crosshair_fake_nomain"))):
    if mod is not None:
        sys.modules[mod.__name__] = mod
    target = "crosshair_fake_does_not_exist" if mod is None else mod.__name__
    try:
        cli.delegate(target, [], _Lg())
        code = 0
    except cli.CLIError as exc:
        code = exc.code
    check(f"delegate to a {name} module -> CLIError(2)", code == 2)
fake = types.ModuleType("crosshair_fake_notimpl")


def _ni(argv):
    raise NotImplementedError("P99")


fake.main = _ni
sys.modules[fake.__name__] = fake
try:
    cli.delegate(fake.__name__, [], _Lg())
    code = 0
except cli.CLIError as exc:
    code = exc.code
check("delegate raising NotImplementedError -> CLIError(2)", code == 2)
fake2 = types.ModuleType("crosshair_fake_exit")
fake2.main = lambda argv: sys.exit(3)
sys.modules[fake2.__name__] = fake2
check("delegate's SystemExit code passes through", cli.delegate(fake2.__name__, [], _Lg()) == 3)

section("train-q / build-neighbours adapters: absent inputs")
import polars as pl                                    # noqa: E402
nohpre = TMP / "valued_no_hpre.parquet"
pl.DataFrame({"action_id": ["a"], "V": [0.1]}).write_parquet(nohpre)
for args, want in ((("train-q",), "--valued"),
                   (("train-q", "--valued", str(TMP / "missing.parquet")), "not found"),
                   (("train-q", "--valued", str(nohpre)), "h_pre"),
                   (("build-neighbours",), "--valued"),
                   (("build-neighbours", "--valued", str(nohpre)), "h_pre")):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main([*args, "--split-file", str(TMP / "splits_v1.json"), *LOGS])   # split required (MED-38)
    check(f"{' '.join(a for a in args if not a.startswith(str(TMP)))} -> exit 2 naming '{want}'",
          rc == 2 and want in err.getvalue() and "Traceback" not in err.getvalue(), err.getvalue()[-200:])
withh = TMP / "valued_hpre.parquet"
pl.DataFrame({"action_id": ["a"], "V": [0.1], "h_pre": [[0.1, 0.2]]}).write_parquet(withh)
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    rc = cli.main(["build-neighbours", "--valued", str(withh), "--q-model", str(TMP / "no_q"),
                   "--split-file", str(TMP / "splits_v1.json"), *LOGS])
check("build-neighbours with a missing Q model dir -> exit 2", rc == 2 and "Q model directory not found" in err.getvalue())

section("eval / sanity / review wiring")
seen = []
real_delegate = cli.delegate
cli.delegate = lambda module, argv, lg: seen.append((module, list(argv))) or 0
with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    cli.main(["eval", "--anchors", "--robustness", "--model", "wp_v1", *LOGS])
    cli.main(["sanity", "--counterfactual", "--model", "wp_v1", *LOGS])
    cli.main(["review", "--n", "5", *LOGS])
cli.delegate = real_delegate
mods = dict(seen)
check("eval --anchors -> engine.evaluation.anchors with --model-dir",
      mods.get("engine.evaluation.anchors", [None])[0] == "--model-dir")
check("eval --robustness -> engine.evaluation.robustness without a model flag",
      mods.get("engine.evaluation.robustness") == [])
check("sanity --counterfactual -> engine.counterfactual.sanity with --model",
      mods.get("engine.counterfactual.sanity", [None])[0] == "--model")
check("review -> engine.evaluation.review with passthrough args", mods.get("engine.evaluation.review") == ["--n", "5"])

section("unexpected exception")


def boom(*a):
    raise RuntimeError("kaboom at https://example.com/x.dem?sig=SECRET")


orig = cli.HANDLERS["status"]
cli.HANDLERS["status"] = boom
for debug in (False, True):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main(["status", *LOGS] + (["--debug"] if debug else []))
    e = err.getvalue()
    check(f"unexpected error exits 1 (debug={debug})", rc == 1)
    check(f"error line names the exception (debug={debug})", "RuntimeError: kaboom" in e)
    check(f"the signed URL query is redacted (debug={debug}; LOW-03)", "SECRET" not in e)
    if debug:
        check("--debug prints the traceback", "Traceback" in e)
    else:
        check("normal mode prints no traceback", "Traceback" not in e)

section("exit codes are 0/1/2 only (LOW-04)")


def ctrl_c(*a):
    raise KeyboardInterrupt


for label, handler, want in (("Ctrl-C", ctrl_c, 1), ("a handler returning 3", lambda *a: 3, 1),
                             ("a handler returning 130", lambda *a: 130, 1), ("a handler returning None", lambda *a: None, 0)):
    cli.HANDLERS["status"] = handler
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main(["status", *LOGS])
    check(f"{label} -> exit {want}", rc == want, f"{rc} {err.getvalue()[-150:]}")
(TMP / "bench").mkdir()
(TMP / "bench" / "bench_value.py").write_text("import sys\nsys.exit(5)\n", encoding="utf-8")
cli.paths.ROOT = TMP
with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    rc = cli.main(["bench", "value", *LOGS])
cli.paths.ROOT = real_root
check("a bench script exiting 5 -> exit 1", rc == 1, str(rc))

section("error handler survives a broken engine.analyze import (LOW-08)")
saved_mod = sys.modules.get("engine.analyze")
sys.modules["engine.analyze"] = None                  # makes `from engine.analyze import ...` raise ImportError
cli.HANDLERS["status"] = boom
err = io.StringIO()
try:
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main(["status", *LOGS])
    raised = None
except Exception as exc:                              # the old handler re-raised from inside `except`
    rc, raised = None, exc
finally:
    if saved_mod is None:
        sys.modules.pop("engine.analyze", None)
    else:
        sys.modules["engine.analyze"] = saved_mod
check("unexpected error still exits 1 with one clean line", raised is None and rc == 1
      and err.getvalue().count("[ERROR") == 1 and "Traceback" not in err.getvalue(), f"{raised!r} {err.getvalue()[-200:]}")
cli.HANDLERS["status"] = orig

section("build-states --all names each failure (LOW-07)")
from engine import db  # noqa: E402
cli.paths.PARSED_DIR = TMP / "parsed"
(TMP / "parsed" / "q_1").mkdir(parents=True)
(TMP / "parsed" / "q_1" / "manifest.json").write_text("{}", encoding="utf-8")
db.register_match({"match_key": "q_1", "demo_sha256": "ab" * 32, "source": "local", "tier": "pro",
                   "status": "quarantined", "quarantine_reason": "tick_rate_64"})
err = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    rc = cli.main(["build-states", "--all", *LOGS])
check("the CLIError message reaches the log, then the count", rc == 1 and "q_1" in err.getvalue()
      and "quarantined" in err.getvalue() and "failed=1" in err.getvalue(), err.getvalue()[-300:])

section("scrape --source hf needs --repo and forwards it (LOW-54)")
real_free, real_dotenv, real_run = cli.paths.free_space_gb, cli._load_dotenv, None
cli.paths.free_space_gb = lambda path=None: 500.0
cli._load_dotenv = lambda path: None                  # never read the real .env in a test
saved_key = os.environ.pop("FACEIT_API_KEY", None)
from engine import ingest  # noqa: E402
real_run, got = ingest.run, {}
ingest.run = lambda source, max_matches=None, **kw: got.update(source=source, **kw) or 0
try:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main(["scrape", "--source", "hf", *LOGS])
    check("--source hf without --repo -> exit 2 naming --repo", rc == 2 and "--repo" in err.getvalue()
          and not got, err.getvalue()[-200:])
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        rc = cli.main(["scrape", "--source", "hf", "--repo", "owner/name", "--max", "1", *LOGS])
    check("--repo reaches the source kwargs; no FACEIT key needed for hf",
          rc == 0 and got.get("source") == "hf" and got.get("repo") == "owner/name", str(got))
finally:
    ingest.run, cli.paths.free_space_gb, cli._load_dotenv = real_run, real_free, real_dotenv
    if saved_key is not None:
        os.environ["FACEIT_API_KEY"] = saved_key

section("bad config")
bad = TMP / "bad.yaml"
bad.write_text("sampling: {sample_hz: 3}\n", encoding="utf-8")
p = run("status", "--config", str(bad), *LOGS)
check("invalid config exits 2 without traceback", p.returncode == 2 and no_tb(p), p.stderr[-200:])

sys.exit(summary())
