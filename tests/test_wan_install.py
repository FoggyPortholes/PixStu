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
