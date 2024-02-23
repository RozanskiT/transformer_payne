class JAXWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ConfigurationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
