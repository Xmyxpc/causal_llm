from result_writer import save_result


result = {
    "model": "Qwen2.5-7B-Instruct",
    "seed": 42,
    "temperature": 0.0,
    "json_valid": True
}


save_result(
    result=result,
    result_dir="results"
)