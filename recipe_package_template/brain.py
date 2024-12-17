from recipe_package_template import Logger

logger = Logger().logger


class Brain:
    """Main class to be replaced

    Attributes
    ----------
    replace_me : str
        Dummy variable to replace
    """

    def __init__(self, replace_me: str):
        """Initializes the sentence encoder and creates output file

        Parameters
        ----------
        replace_me : str
            Dummy variable to replace
        """

        self.replace_me = replace_me

    def __call__(self):
        logger.info("Brain called!")
