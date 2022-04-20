from .context import PythonClueContext

with PythonClueContext(1024, 1024) as ctx:
    ctx.test_binding()