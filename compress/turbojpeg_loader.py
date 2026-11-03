"""TurboJPEG handle for cm=1 that prefers the libjpeg-turbo built by install/setup_linux.sh."""

import os
import sys

from turbojpeg import TurboJPEG

_VENV_LIBJPEG = os.path.join(sys.prefix, "libjpeg-turbo", "lib", "libturbojpeg.so.0")


def open_turbojpeg() -> TurboJPEG:
    """Return a TurboJPEG instance bound to the venv build when it exists.

    PyTurboJPEG 2.x needs libjpeg-turbo 3.x. find_library() picks the system copy first,
    which is often 2.x, so the venv build is passed by explicit path.
    """
    if os.path.exists(_VENV_LIBJPEG):
        return TurboJPEG(_VENV_LIBJPEG)
    return TurboJPEG()
