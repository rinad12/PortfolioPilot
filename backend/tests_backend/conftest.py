import os
import pytest

# Will hold results for reporting at the end
_TEST_RESULTS = []


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
    Writes collected results into reports/test_results.txt.
    """
    os.makedirs("reports", exist_ok=True)

    output_path = os.path.join("reports", "test_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("TEST RESULTS\n")
        f.write("====================\n\n")
        for line in _TEST_RESULTS:
            f.write(line + "\n")

        f.write("\nOverall exit status: " + str(exitstatus))

    print(f"\nTest results written to: {output_path}")
