def format_num(num: int, bytes=False):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def format_time(seconds):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0

    time_us = seconds * 1e6
    if time_us >= US_IN_SECOND:
        return "{:.3f} s".format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return "{:.3f} ms".format(time_us / US_IN_MS)
    return "{:.3f} us".format(time_us)
