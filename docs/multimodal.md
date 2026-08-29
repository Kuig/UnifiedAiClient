# Multimodal Input

> [!NOTE]
> **Placeholder.** This file is not written yet. The content below describes what belongs
> here; the [README](../README.md) still holds it for now.

How files are passed to a model and what each provider accepts. Expected contents, moved
from the README's `### Multimodal Input (Files)` section:

- Passing one file or several through `file_path`, and what the library does with them
  (encoding, upload, cleanup) so that the caller does not have to.
- **The per-provider support matrix**: which providers accept image, audio, video and text
  input. This table is deep-linked from other repositories.
- What counts as a text file, and how a file the provider cannot accept is reported
  (`UnsupportedFileError` and its message format).
