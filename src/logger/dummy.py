"""
Dummy writer that does nothing.
Useful for local debugging without WandB.
"""


class DummyWriter:
    """A no-op writer that mimics the WandBWriter interface."""

    def __init__(self, *args, **kwargs):
        """Accept any arguments but do nothing."""
        pass

    def set_step(self, step, mode="train"):
        """Set current step (no-op)."""
        pass

    def add_scalar(self, name, value, *args, **kwargs):
        """Add scalar (no-op)."""
        pass

    def add_scalars(self, name, values, *args, **kwargs):
        """Add scalars (no-op)."""
        pass

    def add_image(self, name, image, *args, **kwargs):
        """Add image (no-op)."""
        pass

    def add_audio(self, name, audio, *args, **kwargs):
        """Add audio (no-op)."""
        pass

    def add_text(self, name, text, *args, **kwargs):
        """Add text (no-op)."""
        pass

    def add_histogram(self, name, values, *args, **kwargs):
        """Add histogram (no-op)."""
        pass

    def add_table(self, name, table, *args, **kwargs):
        """Add table (no-op)."""
        pass

    def finish(self):
        """Finish logging (no-op)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

