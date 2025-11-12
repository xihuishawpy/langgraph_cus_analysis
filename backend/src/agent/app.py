
import pathlib
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

# 定义 FastAPI 应用
app = FastAPI()


def create_frontend_router(build_dir="../frontend/dist"):
    """创建用于服务 React 前端的路由器。

    参数:
        build_dir: 相对于当前文件的 React 构建目录路径。

    返回:
        用于托管前端的 Starlette 应用。
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # 当前端尚未构建时返回占位路由
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "前端尚未构建，请在 frontend 目录运行 'npm run build'。",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# 将前端挂载在 /app，避免与 LangGraph API 路由冲突
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
