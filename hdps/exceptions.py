import logging

logger = logging.getLogger()
logger.setLevel('ERROR')


class HdpsError(Exception):
    def __init__(self, message: str):
        self.message = message
        logging.error(self.message)
        super().__init__(self.message)


class DuplicateIdError(HdpsError):
    default_message = "There are duplicate IDs in the ID-Column"

    def __init__(self, message: str = default_message):
        self.message = message
        super().__init__(self.message)


class ColumnNotBinaryError(HdpsError):
    default_message = "Treatment column and outcome column (converted outcome column when outcome is continuous) must" \
                      " be binary and contain both 0 and 1"

    def __init__(self, message: str = default_message, column_name: str = None, column_values: list = None):
        self.message = message

        if column_name:
            self.message = f"{self.message} but column {column_name}"
        else:
            self.message = f"{self.message} but this column"

        if column_values:
            self.message = f"{self.message} contains {column_values}"
        else:
            self.message = f"{self.message} contains other values"

        super().__init__(self.message)


class InvalidThresholdValue(HdpsError):
    default_message = "Threshold value is invalid. Must be int, float, or str: ('median', '75p')"

    def __init__(self, message: str = default_message, threshold_value=None):
        if threshold_value:
            self.message = f"{message}. Threshold given: {threshold_value}"
        else:
            self.message = message

        super().__init__(self.message)
