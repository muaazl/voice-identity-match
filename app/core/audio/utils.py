import onnxruntime as ort

def make_session_options(num_threads: int) -> ort.SessionOptions:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return sess_options
