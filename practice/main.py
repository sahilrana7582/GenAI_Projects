from langchain_core.runnables import Runnable, RunnableConfig

from typing import Any, Optional
import time


class DoSomething(Runnable[str, str]):

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> str:

        time.sleep(2)
        return input.upper() + "!"


do = DoSomething()

inputs = [
    "hello",
    "world",
    "python",
    "langchain",
    "runnable",
    "batch",
    "invoke",
    "test",
    "one",
    "two",
]


# -------------------------
# Sequential invoke()
# -------------------------

start = time.perf_counter()

results = []

for item in inputs:
    results.append(do.invoke(item))

end = time.perf_counter()

invoke_time = end - start

print("invoke() results:", results)
print(f"invoke() time: {invoke_time:.2f} seconds")


# -------------------------
# Concurrent batch()
# -------------------------

start = time.perf_counter()

results = do.batch(inputs)

end = time.perf_counter()

batch_time = end - start

print("batch() results:", results)
print(f"batch() time: {batch_time:.2f} seconds")


# -------------------------
# Comparison
# -------------------------

print("\n------ BENCHMARK ------")

print(f"Sequential invoke(): {invoke_time:.2f}s")
print(f"Concurrent batch():  {batch_time:.2f}s")

print(f"Speedup: {invoke_time / batch_time:.2f}x")