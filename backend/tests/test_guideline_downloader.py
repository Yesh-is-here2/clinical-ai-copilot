import hashlib

import httpx
import pytest

from app.retrieval.downloader import (
    GuidelineDownloadError,
    download_guideline,
)
from app.retrieval.sources import GuidelineSource
from app.schemas.document import ClinicalDocumentCategory


def make_source(
    *,
    source_url: str = "https://example.org/guideline",
    ingestion_method: str = "html",
    allowed_hosts: tuple[str, ...] = ("example.org",),
) -> GuidelineSource:
    return GuidelineSource(
        document_id="test-guideline",
        title="Test Guideline",
        organization="Example Organization",
        source_url=source_url,
        publication_date=None,
        category=ClinicalDocumentCategory.GENERAL,
        version="test",
        license_or_usage_note="Test fixture.",
        ingestion_method=ingestion_method,
        allowed_hosts=allowed_hosts,
    )


@pytest.mark.anyio
async def test_download_html_guideline() -> None:
    content = b"<html><body>Clinical evidence</body></html>"

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
                "content-length": str(len(content)),
            },
            content=content,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        result = await download_guideline(
            make_source(),
            client=client,
        )

    assert result.content == content
    assert result.content_type == "text/html"
    assert result.content_length == len(content)
    assert result.sha256 == hashlib.sha256(
        content
    ).hexdigest()


@pytest.mark.anyio
async def test_download_pdf_guideline() -> None:
    content = b"%PDF-1.7 test document"

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf",
            },
            content=content,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        result = await download_guideline(
            make_source(
                ingestion_method="pdf",
            ),
            client=client,
        )

    assert result.content_type == "application/pdf"


@pytest.mark.anyio
async def test_rejects_unexpected_content_type() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
            },
            content=b"unexpected",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            GuidelineDownloadError,
            match="Unexpected response content type",
        ):
            await download_guideline(
                make_source(),
                client=client,
            )


@pytest.mark.anyio
async def test_rejects_oversized_content_length() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
                "content-length": "1000",
            },
            content=b"x",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            GuidelineDownloadError,
            match="size limit",
        ):
            await download_guideline(
                make_source(),
                client=client,
                max_download_bytes=100,
            )


@pytest.mark.anyio
async def test_rejects_stream_that_exceeds_limit() -> None:
    content = b"x" * 101

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
            },
            content=content,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            GuidelineDownloadError,
            match="size limit",
        ):
            await download_guideline(
                make_source(),
                client=client,
                max_download_bytes=100,
            )


@pytest.mark.anyio
async def test_rejects_empty_document() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
            },
            content=b"",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            GuidelineDownloadError,
            match="empty document",
        ):
            await download_guideline(
                make_source(),
                client=client,
            )


@pytest.mark.anyio
async def test_rejects_non_https_source() -> None:
    with pytest.raises(
        GuidelineDownloadError,
        match="HTTPS",
    ):
        await download_guideline(
            make_source(
                source_url="http://example.org/guideline",
            )
        )


@pytest.mark.anyio
async def test_rejects_unapproved_host() -> None:
    with pytest.raises(
        GuidelineDownloadError,
        match="not allowed",
    ):
        await download_guideline(
            make_source(
                allowed_hosts=("trusted.example.org",),
            )
        )


@pytest.mark.anyio
async def test_rejects_failed_http_status() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            404,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            GuidelineDownloadError,
            match="HTTP 404",
        ):
            await download_guideline(
                make_source(),
                client=client,
            )
