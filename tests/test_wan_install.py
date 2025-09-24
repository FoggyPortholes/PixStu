"""Tests for the Wan2.2 installation helper utilities."""

from chargen import wan_install


def test_summarise_install_failure_handles_http_403():
    output = """
    fatal: unable to access 'https://github.com/Wan-Video/Wan2.2.git/': CONNECT tunnel failed, response 403
    error: subprocess-exited-with-error
    """

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "HTTP 403" in message
    assert "proxy" in message.lower()


def test_summarise_install_failure_handles_missing_git():
    output = "'git' is not recognized as an internal or external command, operable program or batch file."

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "Git does not" in message


def test_summarise_install_failure_defaults_to_last_line():
    output = """
    some unrelated warning
    ERROR: could not determine version
    """

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert message == "ERROR: could not determine version"


def test_summarise_install_failure_handles_name_mismatch():
    output = (
        "Requested wan from git+https://github.com/Wan-Video/Wan2.2.git#egg=wan22 has "
        "inconsistent name: expected 'wan22', but metadata has 'wan'"
    )

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "metadata" in message.lower()
    assert "wan" in message
    assert "wan22" in message


def test_summarise_install_failure_handles_python_version_mismatch():
    output = "Python  3.10.9 (you have 3.10.6)"

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "3.10.9" in message
    assert "3.10.6" in message
    assert "Python version" in message


def test_summarise_install_failure_handles_xformers_prompt():
    output = "Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)"

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "xformers" in message.lower()
    assert "reinstall" in message.lower()


def test_summarise_install_failure_handles_git_clone_without_details():
    output = """
    Running command git clone --filter=blob:none --quiet https://github.com/Wan-Video/Wan2.2.git 'C:/Temp/wan22'
    error: subprocess-exited-with-error
    """

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "git" in message.lower()
    assert "clone" in message.lower()
    assert "verbose" in message.lower()


def test_summarise_install_failure_handles_debug_code_placeholder():
    output = "installed failed. debug code"

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "debug code" in message.lower()
    assert "verbose" in message.lower()


def test_summarise_install_failure_handles_dependency_conflicts():
    output = (
        "ERROR: Cannot install wan22 and torch==2.1.0 because these package versions have conflicting dependencies."
    )

    message = wan_install._summarise_install_failure(output)  # noqa: SLF001 - testing helper

    assert "conflict" in message.lower()
    assert "dependency" in message.lower()
