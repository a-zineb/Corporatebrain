import ast
import unittest
from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SOURCE)


class StartupIsolationTests(unittest.TestCase):
    def test_sync_is_not_called_at_module_scope(self):
        top_level_calls = [
            node for node in APP_TREE.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "sync_local_folder_v2"
        ]
        self.assertEqual(top_level_calls, [])

    def test_startup_sync_is_strictly_opt_in(self):
        self.assertIn('ENABLE_STARTUP_SYNC = _env_flag("ENABLE_STARTUP_SYNC")', APP_SOURCE)
        self.assertIn("def maybe_run_startup_sync():", APP_SOURCE)
        self.assertIn("if ENABLE_STARTUP_SYNC:", APP_SOURCE)

    def test_sync_supports_isolated_injected_resources(self):
        self.assertIn("def sync_local_folder_v2(storage_dir=None, target_collection=None, target_embedding_model=None):", APP_SOURCE)
        self.assertIn("target_collection = target_collection or collection", APP_SOURCE)
        self.assertIn("target_embedding_model = target_embedding_model or embedding_model", APP_SOURCE)
        self.assertIn("storage_dir = storage_dir or STORAGE_DIR", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
