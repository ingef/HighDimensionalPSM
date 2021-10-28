import logging


class HdpsError(Exception):
    def __init__(self, message: str):
        self.message = message
        logging.error(self.message)
        super().__init__(self.message)


class DuplicateIdError(HdpsError):
    def __init__(self, message: str):
        super().__init__(message)


class ColumnNotBinaryError(HdpsError):
    def __init__(self, message: str):
        super().__init__(message)


class InvalidThresholdValueError(HdpsError):
    def __init__(self, message: str):
        super().__init__(message)


class ColumnsNotBinaryDueToThresholdError(HdpsError):
    def __init__(self, message: str):
        super().__init__(message)
