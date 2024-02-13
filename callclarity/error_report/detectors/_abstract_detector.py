from abc import ABC, abstractmethod

class AbstractDetector:

    @abstractmethod
    def detect(self):
        pass

    @property
    @abstractmethod
    def _error_flag_col(self):
        pass