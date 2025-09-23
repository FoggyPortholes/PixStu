from typing import Sequence


EXPECTED_TABS = ("Character Studio", "Reference Gallery")


def assert_tabs(tab_labels: Sequence[str]) -> None:
    labels = tuple(tab_labels)
    if labels != EXPECTED_TABS:
        raise ValueError(
            "UI drift detected: expected tabs %s but received %s" % (EXPECTED_TABS, labels)
        )
