import os
import pytest
from datetime import datetime

# Will hold results for reporting at the end
_TEST_RESULTS = []
_TEST_START_TIME = None


def pytest_sessionstart(session):
    """
    Runs at the start of the test session.
    Records the start time for the test run.
    """
    global _TEST_START_TIME
    _TEST_START_TIME = datetime.now()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    This hook is executed for each test phase (setup/call/teardown).
    We want the result of the main "call" phase.
    """
    outcome = yield
    result = outcome.get_result()

    if result.when == "call":
        _TEST_RESULTS.append(
            f"{item.nodeid} - {result.outcome.upper()}"
        )


def pytest_sessionfinish(session, exitstatus):
    """
    Runs after the whole test session finishes.
    Writes collected results into reports with timestamp and test module name.
    """
    os.makedirs("reports", exist_ok=True)

    # Get the test module name from the session
    # Collect unique test module names
    test_modules = set()
    for item in session.items:
        # Extract module name from item.nodeid (e.g., "tests_backend/test_data_pydentic.py::...")
        module_path = item.nodeid.split("::")[0]
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        test_modules.add(module_name)

    # Create timestamp string
    timestamp = _TEST_START_TIME.strftime("%Y-%m-%d_%H-%M-%S")

    # If multiple modules, use a generic name
    if len(test_modules) > 1:
        filename = f"test_results_{timestamp}.txt"
    else:
        module_name = list(test_modules)[0] if test_modules else "tests"
        filename = f"{module_name}_{timestamp}.txt"

    output_path = os.path.join("reports", filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test started: {_TEST_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for line in _TEST_RESULTS:
            f.write(line + "\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Overall exit status: {exitstatus}\n")

        # Summary statistics
        passed = sum(1 for line in _TEST_RESULTS if "PASSED" in line)
        failed = sum(1 for line in _TEST_RESULTS if "FAILED" in line)
        skipped = sum(1 for line in _TEST_RESULTS if "SKIPPED" in line)

        f.write(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped, {len(_TEST_RESULTS)} total\n")

    print(f"\nTest results written to: {output_path}")
