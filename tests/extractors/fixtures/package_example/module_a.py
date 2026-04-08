"""Module A."""

from .module_b import Helper


class Worker:
    """Worker class."""

    def __init__(self, helper: Helper):
        self.helper = helper

    def run(self) -> None:
        self.helper.process()
