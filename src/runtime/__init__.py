"""Frame ingestion: where frames come from, which ones we analyse, and how
backpressure is handled. Nothing here knows about detection or storage.
"""

from .context import Detection, FrameContext  # noqa: F401
from .reader import BUFFER_ALL, DROP_OLDEST, FrameReader  # noqa: F401
from .sampler import FpsSampler  # noqa: F401
from .source import SourceInfo, open_source  # noqa: F401
