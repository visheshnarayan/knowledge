import os
import json
import subprocess
import time
import shutil
import unittest

# --- Test Configuration ---
CONFIG_PATH = "config.yaml"
CONFIG_BACKUP_PATH = "config.yaml.bak"
ORG_TABLE_PATH = "data/builds/knowledge-strands-reasoning/org_table.json"
TEST_TIMEOUT = 20  # seconds to let the main script run

# --- Dummy Data ---
DUMMY_ORG_TABLE = {
    "TestAgent1": {
        "description": "An agent for testing purposes.",
        "keywords": ["test", "dummy"],
    },
    "TestAgent2": {
        "description": "Another agent for testing.",
        "keywords": ["test2", "dummy2"],
    },
}


class TestOrgTableLoading(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        print("--- Setting up test environment for TestOrgTableLoading ---")

        # 1. Create dummy org_table.json
        print(f"Creating dummy org table at: {ORG_TABLE_PATH}")
        os.makedirs(os.path.dirname(ORG_TABLE_PATH), exist_ok=True)
        with open(ORG_TABLE_PATH, "w") as f:
            json.dump(DUMMY_ORG_TABLE, f, indent=4)

        # 2. Backup and modify config.yaml
        print(f"Backing up {CONFIG_PATH} to {CONFIG_BACKUP_PATH}")
        if os.path.exists(CONFIG_PATH):
            shutil.copy(CONFIG_PATH, CONFIG_BACKUP_PATH)

        print("Modifying config.yaml to enable loading from file...")
        with open(CONFIG_PATH, "r") as f:
            lines = f.readlines()

        with open(CONFIG_PATH, "w") as f:
            for line in lines:
                if "load_from_file" in line:
                    f.write("  load_from_file: true\n")
                else:
                    f.write(line)

        # Give the filesystem a moment to catch up
        time.sleep(1)

        print("Setup complete.")

    def tearDown(self):
        """Clean up the test environment after each test."""
        print("\n--- Tearing down test environment for TestOrgTableLoading ---")

        # 1. Restore config.yaml
        print(f"Restoring {CONFIG_PATH} from backup.")
        if os.path.exists(CONFIG_BACKUP_PATH):
            shutil.move(CONFIG_BACKUP_PATH, CONFIG_PATH)

        # 2. Delete dummy org_table.json
        print(f"Deleting dummy org table: {ORG_TABLE_PATH}")
        if os.path.exists(ORG_TABLE_PATH):
            os.remove(ORG_TABLE_PATH)

        print("Teardown complete.")

    def test_loading_from_file(self):
        """
        Tests if the application correctly loads a pre-existing OrgTable
        when the configuration is set to do so.
        """
        print("\n--- Running test: test_loading_from_file ---")
        print(f"Starting main.py. The test will run for {TEST_TIMEOUT} seconds...")

        process = subprocess.Popen(
            ["uv", "run", "python3", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = ""
        try:
            for line in iter(process.stdout.readline, ""):
                # Don't print live output to keep test results clean
                output += line
                if "Flask app created." in line:
                    time.sleep(2)
                    break

            time.sleep(TEST_TIMEOUT - 2)

        finally:
            print(
                f"\n--- Test duration ({TEST_TIMEOUT}s) finished. Terminating process. ---"
            )
            process.terminate()
            try:
                remaining_output, _ = process.communicate(timeout=5)
                output += remaining_output
            except subprocess.TimeoutExpired:
                process.kill()
                print("Process was killed as it did not terminate gracefully.")

        print("\n--- Verifying results ---")

        expected_log = f"Organizational table loaded from {ORG_TABLE_PATH}"
        self.assertIn(
            expected_log,
            output,
            f"Expected log message not found in output: '{expected_log}'",
        )

        unexpected_log = "Populating organizational table..."
        self.assertNotIn(
            unexpected_log,
            output,
            "Found unexpected message: 'Populating organizational table...'",
        )


if __name__ == "__main__":
    unittest.main()
