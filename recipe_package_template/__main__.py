import argparse
from recipe_package_template.brain import Brain
from recipe_package_template import Logger

logger = Logger().logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse the arguments that start the process

    Returns
    -------
    The arguments of the parser
    """

    parser = argparse.ArgumentParser(
        description="Command Line Interface for bootstrap project"
    )

    subparsers = parser.add_subparsers(
        description="Use different parser to distinguish between batch and single mode.",
        dest="mode",
    )

    mode1 = subparsers.add_parser(name="mode1", help="Sample mode1")

    mode2 = subparsers.add_parser(
        name="mode2",
        help="Sample mode2",
    )

    mode1.add_argument(
        "-arg1",
        type=str,
        required=True,
        help="Sample argument.",
    )

    mode2.add_argument(
        "-arg2",
        type=str,
        required=True,
        help="Sample argument",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    match arguments.mode:
        case "mode1":
            _ = Brain(replace_me=arguments.arg1)
            logger.info("Completed")

        case "mode2":
            _ = Brain(replace_me=arguments.arg2)
        case _:
            logger.error("Mode not recognised. Try one of: mode1, mode2")


if __name__ == "__main__":
    main()
