from fastapi import APIRouter, Depends

from ..deps import Container, get_container
from ..schemas import MethodView

router = APIRouter(tags=["methods"])


@router.get("/methods", response_model=list[MethodView])
def list_methods(container: Container = Depends(get_container)) -> list[MethodView]:
    return [MethodView(name=name, default_weight=weight) for name, weight in container.method_weights.items()]
