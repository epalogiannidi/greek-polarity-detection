import logging.config
import os
import time

from greek_polarity_detection.about import __title__


class Logger:
    def __init__(self):

        basename = os.path.basename(os.getcwd())
        rootdir = os.path.dirname(os.getcwd())

        fname = (
            os.path.join(rootdir, "configs/logger.conf")
            if basename in ["docs", f"{__title__.replace('-', '_')}"]
            else os.path.join(os.getcwd(), "configs/logger.conf")
        )
        logdir = os.path.join(rootdir, "logs")
        logname = time.strftime("%m%d%Y-%H:%M:%S")

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logging.config.fileConfig(
            fname=fname,
            disable_existing_loggers=False,
            defaults={"logfilename": f"{logdir}/{logname}.log"},
        )
        logger = logging.getLogger(__name__)
        if basename == "docs":
            logging.disable(logging.CRITICAL)
            logger = logging.getLogger(__name__)
        else:
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            logging.config.fileConfig(
                fname=fname,
                disable_existing_loggers=False,
                defaults={"logfilename": f"{logdir}/{logname}.log"},
            )

            logger = logging.getLogger(__name__)
        self.logger = logger
