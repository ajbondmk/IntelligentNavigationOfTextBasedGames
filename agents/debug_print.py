DEBUG = True

def debug_print(string = ""):
    if DEBUG:
        print(string)

def debug_not_print(string = ""):
    if not DEBUG:
        print(string)
