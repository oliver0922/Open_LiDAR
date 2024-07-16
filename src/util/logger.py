from colorama import init, Fore, Back, Style
import os 
import sys

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def print_bright(s):

    init()
    print(Style.BRIGHT + s + Style.RESET_ALL)


def print_green(info, value=""):

    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value=""):

    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def str_to_bluestr(string):

    return Fore.BLUE + "%s" % string + Style.RESET_ALL


def str_to_yellowstr(string):

    return Fore.YELLOW + "%s" % string + Style.RESET_ALL