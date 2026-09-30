from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status

from ..deps import Container, get_container
from ..schemas import DatasetDetails, DatasetSummary, ProfileRequest

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _summary(entry) -> DatasetSummary:
    return DatasetSummary(
        id=entry.id,
        name=entry.name,
        description=entry.description,
        origin=entry.origin,
        default_max_lag=entry.default_max_lag,
    )


@router.get("", response_model=list[DatasetSummary])
def list_datasets(container: Container = Depends(get_container)) -> list[DatasetSummary]:
    return [_summary(entry) for entry in container.datasets.list()]


@router.post("", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    date_column: Annotated[str | None, Form()] = None,
    container: Container = Depends(get_container),
) -> DatasetSummary:
    content = await file.read()
    name = (file.filename or "upload.csv").rsplit(".", 1)[0]
    return _summary(container.datasets.upload(name, content, date_column or None))


@router.get("/{dataset_id}", response_model=DatasetDetails)
def dataset_details(dataset_id: str, container: Container = Depends(get_container)) -> dict[str, Any]:
    return container.datasets.details(dataset_id)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, container: Container = Depends(get_container)) -> Response:
    container.datasets.delete_upload(dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{dataset_id}/profile")
def profile_dataset(
    dataset_id: str,
    body: ProfileRequest,
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    return container.datasets.profile(dataset_id, body.columns, body.declared_causal_sufficiency)
