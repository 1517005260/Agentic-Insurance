"""Collection override: invariant tests live in ``pN_*.py`` not ``test_*.py``.

The paper appendix B pins the file naming as ``tests/invariants/pN_*.py``
so the per-protocol files are addressable by name in references. We
override ``python_files`` for this directory only — global pytest config
still requires the ``test_*.py`` convention everywhere else.
"""

collect_ignore_glob = []


def pytest_collect_file(parent, file_path):
    # Defer to pytest's default collection but widen the filename
    # pattern. ``pytest_collect_file`` returns ``None`` to fall back to
    # the chain; we just need to override the discovery pattern via
    # a configure hook.
    return None


def pytest_configure(config):
    # Append the pN pattern so files like p1_citation_stability.py
    # collect as ``test_*`` modules do (they expose ``test_*`` funcs).
    pf = list(config.getini("python_files"))
    if "p?_*.py" not in pf:
        pf.append("p?_*.py")
        # ``addini``'s underlying setting expects a space-separated str
        # for the ``ini`` reading path; mutating the list in place is
        # the supported pattern (see pytest docs ``python_files``).
        config._inicache["python_files"] = pf
