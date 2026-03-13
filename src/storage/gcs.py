import os
from pathlib import Path
from google.cloud import storage


def _list_blobs(client: storage.Client, bucket_name: str, prefix: str):
    bucket = client.bucket(bucket_name)
    return list(client.list_blobs(bucket, prefix=prefix))


def download_prefix(bucket_name: str, prefix: str, local_dir: str) -> None:
    """
    Download all blobs under gs://bucket/prefix to local_dir (keeping relative paths).
    """
    client = storage.Client()
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    blobs = _list_blobs(client, bucket_name, prefix)
    if not blobs:
        return

    for blob in blobs:
        # skip "directory marker" objects
        if blob.name.endswith("/"):
            continue

        rel = blob.name[len(prefix):].lstrip("/")  # path relative to prefix
        dest_path = Path(local_dir) / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest_path))


def upload_dir(local_dir: str, bucket_name: str, prefix: str) -> None:
    """
    Upload all files under local_dir to gs://bucket/prefix/...
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_dir_path = Path(local_dir)
    if not local_dir_path.exists():
        raise FileNotFoundError(f"Local dir not found: {local_dir}")

    for path in local_dir_path.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(local_dir_path).as_posix()
        blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
        blob.upload_from_filename(str(path))
