import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from app.retrieval.sources import GuidelineSource
from app.schemas.ingestion import DownloadedGuideline


DEFAULT_MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0

HTML_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
}

PDF_CONTENT_TYPES = {
    "application/pdf",
}


class GuidelineDownloadError(RuntimeError):
    pass


def _normalized_content_type(value: str | None) -> str:
    if not value:
        return ""

    return value.split(";", maxsplit=1)[0].strip().lower()


def _validate_https_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme.lower() != "https":
        raise GuidelineDownloadError(
            "Guideline sources must use HTTPS."
        )

    if not parsed.hostname:
        raise GuidelineDownloadError(
            "Guideline source URL does not contain a valid hostname."
        )


def _validate_host(
    url: str,
    allowed_hosts: tuple[str, ...],
) -> None:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    allowed = {
        host.lower()
        for host in allowed_hosts
    }

    if hostname not in allowed:
        raise GuidelineDownloadError(
            f"Host '{hostname}' is not allowed for this guideline source."
        )


def _allowed_content_types(
    ingestion_method: str,
) -> set[str]:
    normalized_method = ingestion_method.strip().lower()

    if normalized_method == "html":
        return HTML_CONTENT_TYPES

    if normalized_method == "pdf":
        return PDF_CONTENT_TYPES

    raise GuidelineDownloadError(
        f"Unsupported ingestion method '{ingestion_method}'."
    )


async def download_guideline(
    source: GuidelineSource,
    *,
    client: httpx.AsyncClient | None = None,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> DownloadedGuideline:
    if max_download_bytes <= 0:
        raise ValueError(
            "max_download_bytes must be greater than zero."
        )

    source_url = str(source.source_url)

    _validate_https_url(source_url)
    _validate_host(
        source_url,
        source.allowed_hosts,
    )

    allowed_content_types = _allowed_content_types(
        source.ingestion_method
    )

    owns_client = client is None

    if client is None:
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            headers={
                "User-Agent": (
                    "Clinical-AI-Copilot-Educational-Research/0.1"
                )
            },
        )

    try:
        async with client.stream(
            "GET",
            source_url,
        ) as response:
            response.raise_for_status()

            final_url = str(response.url)

            _validate_https_url(final_url)
            _validate_host(
                final_url,
                source.allowed_hosts,
            )

            content_type = _normalized_content_type(
                response.headers.get("content-type")
            )

            if content_type not in allowed_content_types:
                raise GuidelineDownloadError(
                    "Unexpected response content type "
                    f"'{content_type or 'missing'}' for "
                    f"ingestion method '{source.ingestion_method}'."
                )

            header_length = response.headers.get(
                "content-length"
            )

            if header_length:
                try:
                    declared_length = int(header_length)
                except ValueError as exc:
                    raise GuidelineDownloadError(
                        "Invalid Content-Length header."
                    ) from exc

                if declared_length > max_download_bytes:
                    raise GuidelineDownloadError(
                        "Guideline document exceeds the configured "
                        "download size limit."
                    )

            payload = bytearray()

            async for chunk in response.aiter_bytes():
                payload.extend(chunk)

                if len(payload) > max_download_bytes:
                    raise GuidelineDownloadError(
                        "Guideline document exceeds the configured "
                        "download size limit."
                    )

    except httpx.HTTPStatusError as exc:
        raise GuidelineDownloadError(
            f"Guideline source returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise GuidelineDownloadError(
            "Guideline source request failed."
        ) from exc
    finally:
        if owns_client:
            await client.aclose()

    if not payload:
        raise GuidelineDownloadError(
            "Guideline source returned an empty document."
        )

    content = bytes(payload)

    return DownloadedGuideline(
        document_id=source.document_id,
        source_url=source.source_url,
        final_url=final_url,
        content_type=content_type,
        content_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        downloaded_at=datetime.now(timezone.utc),
        content=content,
    )
