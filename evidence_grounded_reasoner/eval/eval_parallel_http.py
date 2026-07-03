import json
import os
import re
import tempfile
import time
from argparse import ArgumentParser
from datetime import datetime
import multiprocessing as mp

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

TAG_THINK = 'think'
TAG_RETHINK = 'rethink'
TAG_ANSWER = 'answer'


def parse_arguments():
    parser = ArgumentParser(description='Parallel HTTP inference for VL model evaluation')
    parser.add_argument('-b', '--benchmark-file', required=True,
                        help='Benchmark JSON with video_path and question fields')
    parser.add_argument('-o', '--output-file', default='',
                        help='Output JSON path (default: res.json next to benchmark file)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel worker processes')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--ckpt-path', required=True,
                        help='Checkpoint path (recorded in output metadata)')
    parser.add_argument('--heartbeat-file', default='')
    parser.add_argument('--heartbeat-interval', type=int, default=60)
    parser.add_argument('--no-progress-timeout', type=int, default=1800)
    parser.add_argument('--request-connect-timeout', type=int, default=30)
    parser.add_argument('--request-read-timeout', type=int, default=600)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--retry-backoff', type=float, default=2.0)
    parser.add_argument('--system-prompt', default='',
                        help='Optional system prompt to prepend to each request')
    return parser.parse_args()


def load_benchmark_data(benchmark_file: str) -> list:
    print(f"Loading benchmark data from '{benchmark_file}'...")
    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f'Benchmark file not found: {benchmark_file}')
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} samples.')
    return data


def extract_answer(response_text: str) -> str:
    pat = rf'<{TAG_ANSWER}>(.*?)</{TAG_ANSWER}>'
    match = re.search(pat, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    close_tag = f'</{TAG_RETHINK}>'
    idx = response_text.rfind(close_tag)
    if idx != -1:
        return response_text[idx + len(close_tag):].strip()
    return response_text.strip()


def check_format(response_text: str) -> tuple:
    has_think = bool(re.search(rf'<{TAG_THINK}>.*?</{TAG_THINK}>', response_text, re.DOTALL))
    has_rethink = bool(re.search(rf'<{TAG_RETHINK}>.*?</{TAG_RETHINK}>', response_text, re.DOTALL))
    has_answer = bool(re.search(rf'<{TAG_ANSWER}>.*?</{TAG_ANSWER}>', response_text, re.DOTALL))
    return has_think, has_rethink, has_answer, has_think and has_rethink and has_answer


def build_messages(question: str, system_prompt: str = ''):
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': question})
    return messages


def create_session(args) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retries = Retry(
        total=args.max_retries,
        connect=args.max_retries,
        read=args.max_retries,
        status=args.max_retries,
        backoff_factor=args.retry_backoff,
        allowed_methods=frozenset(['GET', 'POST']),
        status_forcelist=[502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def atomic_write_json(path: str, payload: dict):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_dir or None, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_heartbeat(path: str, payload: dict):
    if path:
        atomic_write_json(path, payload)


def worker_process(worker_args):
    worker_id, data_chunk, args = worker_args
    base_url = f'http://{args.host}:{args.port}'
    local_results = []
    session = create_session(args)

    model_response = session.get(
        f'{base_url}/v1/models',
        timeout=(args.request_connect_timeout, args.request_read_timeout),
    )
    model_response.raise_for_status()
    model_id = model_response.json()['data'][0]['id']

    for item in tqdm(data_chunk, desc=f'Worker-{worker_id}', position=worker_id):
        error = ''
        model_raw_output = ''
        prediction = 'ERROR'
        has_think = has_rethink = has_answer = format_ok = False
        try:
            video_path = item['video_path']
            question = item['question']
            if not os.path.exists(video_path):
                raise FileNotFoundError(f'Video file not found: {video_path}')
            payload = {
                'model': model_id,
                'messages': build_messages(question, args.system_prompt),
                'videos': [video_path],
                'max_tokens': 4096,
                'temperature': 0,
                'stream': False,
            }
            response = session.post(
                f'{base_url}/v1/chat/completions',
                json=payload,
                timeout=(args.request_connect_timeout, args.request_read_timeout),
            )
            response.raise_for_status()
            model_raw_output = response.json()['choices'][0]['message']['content']
            prediction = extract_answer(model_raw_output)
            has_think, has_rethink, has_answer, format_ok = check_format(model_raw_output)
        except Exception as exc:
            error = str(exc)
            model_raw_output = f'Inference failed with error: {exc}'
            tqdm.write(f"\n[Worker-{worker_id}] Error processing '{item.get('video_path', 'UNKNOWN')}': {exc}")

        local_results.append({
            'benchmark_index': item.get('_benchmark_index', -1),
            'video_path': item.get('video_path', ''),
            'question': item.get('question', ''),
            'model_raw_output': model_raw_output,
            'prediction': prediction,
            'has_think': has_think,
            'has_rethink': has_rethink,
            'has_answer': has_answer,
            'format_ok': format_ok,
            'error': error,
            'source_item': {k: v for k, v in item.items() if not k.startswith('_')},
        })
    return local_results


def main():
    args = parse_arguments()
    if args.num_workers <= 0:
        raise ValueError('--num-workers must be > 0')
    benchmark_data = load_benchmark_data(args.benchmark_file)
    for idx, item in enumerate(benchmark_data):
        item['_benchmark_index'] = idx
    num_workers = min(args.num_workers, len(benchmark_data)) if benchmark_data else 0
    if num_workers == 0:
        raise ValueError('Benchmark data is empty')

    output_file = args.output_file or os.path.join(os.path.dirname(args.benchmark_file), 'res.json')
    data_chunks = [benchmark_data[i::num_workers] for i in range(num_workers)]
    print(f"\nStarting parallel HTTP inference ({len(benchmark_data)} samples, {num_workers} workers)")
    print(f'Target server: http://{args.host}:{args.port}')

    write_heartbeat(args.heartbeat_file, {
        'ckpt_path': args.ckpt_path,
        'completed': 0,
        'total': len(benchmark_data),
        'errors': 0,
        'last_update_ts': datetime.now().isoformat(timespec='seconds'),
        'status': 'running',
    })

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        all_results_nested = pool.map(
            worker_process,
            [(worker_id, data_chunks[worker_id], args) for worker_id in range(num_workers)],
        )

    results = sorted(
        [item for sublist in all_results_nested for item in sublist],
        key=lambda x: x['benchmark_index'],
    )
    error_samples = sum(1 for item in results if item.get('error'))
    final_output = {
        'summary': {
            'checkpoint_path': args.ckpt_path,
            'benchmark_file': args.benchmark_file,
            'total_samples': len(results),
            'format_ok_samples': sum(1 for item in results if item.get('format_ok')),
            'has_think_samples': sum(1 for item in results if item.get('has_think')),
            'has_rethink_samples': sum(1 for item in results if item.get('has_rethink')),
            'has_answer_samples': sum(1 for item in results if item.get('has_answer')),
            'error_samples': error_samples,
            'gpus_used': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'max_tokens': 4096,
            'max_retries': args.max_retries,
        },
        'results': results,
    }
    atomic_write_json(output_file, final_output)
    write_heartbeat(args.heartbeat_file, {
        'ckpt_path': args.ckpt_path,
        'completed': len(results),
        'total': len(benchmark_data),
        'errors': error_samples,
        'last_update_ts': datetime.now().isoformat(timespec='seconds'),
        'last_progress_ts': datetime.now().isoformat(timespec='seconds'),
        'status': 'completed',
    })
    print(f"\nInference complete. Total: {len(results)}, Errors: {error_samples}")
    print(f'Results saved to: {output_file}')


if __name__ == '__main__':
    main()
