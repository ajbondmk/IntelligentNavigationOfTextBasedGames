"""
If DEBUG is True, only print the strings passed into debug_print.
If DEBUG is False, only print the strings passed into debug_not_print.
"""


DEBUG = False


def debug_print(string = ""):
    if DEBUG:
        print(string)


def debug_not_print(string = ""):
    if not DEBUG:
        print(string)
