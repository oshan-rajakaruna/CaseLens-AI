"""Generate tiny synthetic PDF fixtures; contains no real legal material."""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

FIXTURE_DIR = Path(__file__).parent


def write_synthetic_pdf(path: Path, pages: list[str | None]) -> None:
    """Write controlled synthetic text pages and optional text-free pages."""

    writer = PdfWriter()
    for page_text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if page_text is None:
            continue
        escaped_text = (
            page_text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font}
                )
            }
        )
        contents = DecodedStreamObject()
        contents.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(contents)

    with path.open("wb") as destination:
        writer.write(destination)


def main() -> None:
    write_synthetic_pdf(
        FIXTURE_DIR / "synthetic_valid.pdf",
        ["SYNTHETIC PDF TEST DATA. Not real case law."],
    )
    write_synthetic_pdf(
        FIXTURE_DIR / "synthetic_multipage.pdf",
        [
            "FIRST PAGE synthetic employment dismissal notice section 1.",
            None,
            "THIRD PAGE synthetic hearing procedure section 3.",
        ],
    )
    write_synthetic_pdf(FIXTURE_DIR / "synthetic_no_text.pdf", [None, None])
    write_synthetic_pdf(FIXTURE_DIR / "synthetic_zero_pages.pdf", [])
    (FIXTURE_DIR / "synthetic_corrupted.pdf").write_bytes(
        b"%PDF-1.7\nSYNTHETIC CORRUPTED TEST FIXTURE"
    )


if __name__ == "__main__":
    main()
