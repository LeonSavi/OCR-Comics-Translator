
# https://alexandra-zaharia.github.io/posts/function-timeout-in-python-signal/
# Signal may work only with Linux/MacOS systems
import signal
from typing import NamedTuple,Iterable
# from tqdm import tqdm # TO FIND SOMETHING DIFFERENT FROM TQDM


class Colours(NamedTuple):
    HEADER = "\033[95m"   # purple/magenta
    END = "\033[0m"       # reset


class Timeout(Exception):
    '''Timeout Exception we raise when something wrong happens'''
    pass


class LoopingTime:
    def __init__(self, timeout: int = 20):
        
        self.timeout = timeout

        # assign alarm type and func
        signal.signal(signal.SIGALRM, self._handle_timeout)
    
    def _handle_timeout(self, sig, frame):
        raise Timeout

    def iterate(self,iterable:Iterable, function):
        for i in iterable:
            try:
                signal.alarm(self.timeout) # non zero
                yield function(i) # where the issue might happen
            except Timeout as exc:
                print(f"{Colours.HEADER}Pass iteration {i}: Took too long!{Colours.END}")
                print('{}: {}'.format(exc.__class__.__name__, exc))
            finally:
                signal.alarm(0) #cancel the alarm

