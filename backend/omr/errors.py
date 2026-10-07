class OMRError(Exception):
    """Base class for user-facing OMR errors."""


class InvalidImageError(OMRError):
    """Raised when uploaded bytes cannot be decoded as an image."""


class LayoutDetectionError(OMRError):
    """Raised when the configured answer sheet cannot be located."""

