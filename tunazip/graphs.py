import torch


class GraphCache:
    def __init__(self, enabled):
        self.enabled = enabled
        self.pool = torch.cuda.graph_pool_handle() if enabled else None
        if enabled:
            torch.autograd.graph.set_override_stale_capture_stream(True)
        self.seen = set()
        self.captured = {}

    def call(self, key, inputs, body):
        if not self.enabled or key is None:
            return body(*inputs)
        if key not in self.seen:
            self.seen.add(key)
            return body(*inputs)
        if key not in self.captured:
            static_inputs = tuple(tensor.clone() for tensor in inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                outputs = body(*static_inputs)
            self.captured[key] = (graph, static_inputs, outputs)
        else:
            graph, static_inputs, outputs = self.captured[key]
            for static, tensor in zip(static_inputs, inputs):
                static.copy_(tensor)
        graph.replay()
        return outputs
