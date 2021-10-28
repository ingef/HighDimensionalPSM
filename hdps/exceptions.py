import logging


class HdpsError(Exception):
    def __init__(self, message: str):
        self.message = message
        logging.error(self.message)
        super().__init__(self.message)


class DuplicateIdError(HdpsError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ColumnNotBinaryError(HdpsError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class InvalidThresholdValueError(HdpsError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ColumnsNotBinaryDueToThresholdError(HdpsError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
