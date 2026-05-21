from __future__ import annotations

import json
import os

from evals.vista_sta.ego4d_sta import prebuild_sta_caches


def main() -> None:
    train_root = os.environ["STA_TRAIN_ROOT"]
    val_root = os.environ["STA_VAL_ROOT"]
    test_root = os.environ.get("STA_TEST_ROOT") or None
    build_test = os.environ.get("STA_BUILD_TEST_CACHE", "0").lower() in {"1", "true", "yes"}
    stats = prebuild_sta_caches(train_root, val_root, test_root, build_test)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
