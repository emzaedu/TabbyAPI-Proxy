import os, subprocess, time, threading, requests, json, logging, sys, yaml, hashlib, atexit, io
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
from functools import wraps

DEBUG_OUTPUT = True
FLASK_DEBUG = True

PROXY_HOST = '127.0.0.1'
PROXY_PORT = 9000
DEFAULT_UNLOAD_TIMER = 86400
API_ENDPOINT = "http://127.0.0.1:7001"
CONFIG_DIR = "config"
MODEL_DIR = "models"
TABBY_CONFIG_PATH = "config.yml"
API_TOKENS_PATH = "api_tokens.yml"
ALLOWED_HOSTS = ['api.externaldomain.com']

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG if DEBUG_OUTPUT else logging.INFO)

if os.path.exists(TABBY_CONFIG_PATH):
    with open(TABBY_CONFIG_PATH, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    host = main_config.get('network', {}).get('host', '127.0.0.1')
    port = main_config.get('network', {}).get('port', 5000)
    API_ENDPOINT = f"http://{host}:{port}"
    MODEL_DIR = main_config.get('model', {}).get('model_dir', 'models')
    logging.debug(f"Started process parsed from {TABBY_CONFIG_PATH}: API endpoint {API_ENDPOINT}; model directory: {MODEL_DIR}")
else:
    logging.error("Main configuration file not found.")

if os.path.exists(API_TOKENS_PATH):
    with open(API_TOKENS_PATH, 'r', encoding='utf-8') as f:
        api_tokens = yaml.safe_load(f)
    logging.debug(f"Loaded API tokens from {API_TOKENS_PATH}")
else:
    logging.error("API tokens configuration file not found.")
    api_tokens = {}

@app.before_request
def log_request_info():
    logging.debug(f"Request Headers: {request.headers}")
    logging.debug(f"Request Body: {request.get_data(as_text=True)}")

def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        host_header = request.headers.get('Host')
        if host_header not in ALLOWED_HOSTS:
            return func(*args, **kwargs)
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.error("Authorization header is missing or incorrect.")
            return jsonify({"error": "FORBIDDEN"}), 403
        token = auth_header.split('Bearer ')[1]
        if token not in api_tokens.values():
            logging.error("Invalid API key.")
            return jsonify({"error": "FORBIDDEN"}), 403
        return func(*args, **kwargs)
    return wrapper

class ModelServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelServer, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.default_unload_timer = DEFAULT_UNLOAD_TIMER
        self.unload_timer = DEFAULT_UNLOAD_TIMER
        self.current_model = None
        self.current_process = None
        self.server_ready = False
        self.server_ready_event = threading.Event()
        self.last_access_time = None
        self.process_lock = threading.Lock()
        self.python_executable = sys.executable
        self.model_params = None
        self.draft_model_params = None
        self.model_template = None
        threading.Thread(target=self.check_last_access_time, daemon=True).start()
        atexit.register(self.stop_model_process)

    def start_model_process(self, model):
        if self.current_process is not None:
            self.stop_model_process()
        
        config_path = os.path.join(CONFIG_DIR, model.replace('/', os.sep) + '.yml')
        if os.path.commonpath([os.path.abspath(config_path), os.path.abspath(CONFIG_DIR)]) != os.path.abspath(CONFIG_DIR):
            logging.error(f"Attempted to access forbidden path: {config_path}")
            return False
        if not os.path.exists(config_path):
            logging.error(f"Config file for model {model} not found.")
            return False

        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        self.model_params = model_config.get('model', {})
        self.draft_model_params = model_config.get('draft_model', {})
        self.model_template = model_config.get('template', {})

        model_name = self.model_params.get('model_name')
        if not model_name:
            logging.error(f"Model name not found in config file for model {model}.")
            return False

        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            logging.error(f"Model directory for model {model} not found: {model_path}")
            return False

        self.current_process = subprocess.Popen(
            [self.python_executable, "start.py", "--config", config_path]
        )
        logging.debug(f"Started process for model {model} with PID {self.current_process.pid}")
        self.server_ready = False
        self.server_ready_event.clear()
        threading.Thread(target=self.check_server_ready, daemon=True).start()
        self.last_access_time = time.time()
        return True

    def stop_model_process(self):
        if self.current_process is not None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("Terminate failed. Killing process...")
                self.current_process.kill()
            logging.debug(f"Stopped process with PID {self.current_process.pid}")
            self.current_process = None
            self.server_ready = False
            self.server_ready_event.clear()
            self.last_access_time = None
            self.current_model = None
            self.unload_timer = self.default_unload_timer
            self.model_params = None
            self.draft_model_params = None
            self.model_template = None

    def check_server_ready(self):
        for _ in range(30):
            if self.is_server_ready():
                self.server_ready = True
                self.server_ready_event.set()
                logging.debug("Server is ready.")
                break
            time.sleep(1)
        else:
            logging.error("Server is not ready after 30 seconds. Terminating process.")
            self.stop_model_process()

    def is_server_ready(self):
        try:
            response = requests.get(f"{API_ENDPOINT}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def check_last_access_time(self):
        while True:
            if self.last_access_time is not None:
                elapsed = time.time() - self.last_access_time
                if elapsed > self.unload_timer:
                    logging.info(f"No requests for {elapsed} seconds. Stopping the model process.")
                    self.stop_model_process()
            time.sleep(5)

    def get_status(self):
        model = self.current_model if self.current_model else None
        if self.last_access_time is None:
            until = None
        else:
            elapsed = time.time() - self.last_access_time
            until = max(0, int(self.unload_timer - elapsed))
        if until is not None:
            if until < 60:
                until_str = f"{until}s"
            elif until < 3600:
                minutes = until // 60
                seconds = until % 60
                until_str = f"{minutes}m {seconds}s"
            else:
                hours = until // 3600
                minutes = (until % 3600) // 60
                seconds = until % 60
                until_str = f"{hours}h {minutes}m {seconds}s"
        else:
            until_str = None

        return {
            "model": model,
            "until": until,
            "until_str": until_str,
            "model_params": self.model_params,
            "draft_model_params": self.draft_model_params,
            "model_template": self.model_template
        }

model_server = ModelServer()

def update_from_template(data, template):
    params = [
        'top_p', 'min_p', 'top_k', 'temperature', 'top_a', 'tfs', 'frequency_penalty',
        'mirostat_mode', 'mirostat', 'mirostat_eta', 'mirostat_tau', 'repetition_penalty',
        'presence_penalty', 'repetition_range', "min_tokens", "generate_window", "stop",
        "banned_strings", "token_healing", "temperature_last", "smoothing_factor", "skew",
        "xtc_probability", "xtc_threshold", "dry_multiplier", "dry_base", "dry_allowed_length",
        "dry_range", "dry_sequence_breakers", "add_bos_token", "ban_eos_token",
        "skip_special_tokens", "negative_prompt", "json_schema", "regex_pattern", "grammar_string",
        "speculative_ngram", "cfg_scale", "max_temp", "min_temp", "temp_exponent", "n",
        "best_of", "echo", "suffix", "user", "prompt_template", "add_generation_prompt",
        "response_prefix", 'penalty_range', "stream", "banned_tokens", "allowed_tokens",
        "logit_bias", "stream_options", "logprobs", "response_format", "template_vars",
    ]
    for param in params:
        if param not in data and param in template:
            data[param] = template[param]
            logging.debug(f"Parameter {param} set from template: {data[param]}")

def prepare_model_request(data):
    keep_alive = data.get('keep_alive')
    model_server.unload_timer = keep_alive if keep_alive is not None else model_server.default_unload_timer
    model = data.get('model')
    if model and model != model_server.current_model:
        logging.info(f"Model changed: {model_server.current_model} -> {model}")
        if not model_server.start_model_process(model):
            return False, f"Config file for model {model} not found."
        model_server.current_model = model
    for _ in range(60):
        if model_server.is_server_ready():
            break
        time.sleep(1)
    if not model_server.server_ready_event.wait(timeout=60):
        return False, "Server is not ready"
    model_server.last_access_time = time.time()
    data.pop('model', None)
    data.pop('keep_alive', None)
    return True, None

def inject_system_message(data, template):
    system_time_flag = template.get("system_time", False)
    if system_time_flag:
        now = datetime.now()
        formatted_time = now.strftime('%H:%M:%S %Z%z')
        formatted_date = now.strftime('%Y-%m-%d')
        system_time_str = f"Current time: {formatted_time}\nToday: {formatted_date}"
    else:
        system_time_str = ""
    messages = data.get('messages', [])
    system_instruction = template.get('system')
    if system_time_str:
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, {"role": "system", "content": system_time_str + (("\n\n" + system_instruction) if system_instruction else "")})
        else:
            messages[0] = {"role": "system", "content": system_time_str + "\n\n" + messages[0].get('content', "")}
    else:
        if not messages or messages[0].get('role') != 'system':
            if system_instruction:
                messages.insert(0, {"role": "system", "content": system_instruction})
    data['messages'] = messages

def list_model_ids():
    model_ids = []
    for root, _, files in os.walk(CONFIG_DIR):
        for file in files:
            if file.endswith('.yml'):
                rel_path = os.path.relpath(root, CONFIG_DIR)
                model_id = os.path.splitext(file)[0] if rel_path == '.' else os.path.join(rel_path, os.path.splitext(file)[0]).replace(os.sep, '/')
                model_ids.append(model_id)
    return model_ids

def ollama_update_from_options(openai_data, options):
    for param in ['stop', 'temperature', 'mirostat', 'mirostat_eta', 'mirostat_tau',
                  'top_k', 'top_p', 'min_p', 'frequency_penalty', 'tfs_z', "max_tokens", "num_predict"]:
        if param in options:
            if param == 'top_k':
                logging.debug(f"Original value of {param}: {options[param]}")
                openai_data["top_k"] = int(float(options[param]))
            elif param == 'tfs_z':
                openai_data["tfs"] = options[param]
            elif param == 'num_predict':
                openai_data["max_tokens"] = options[param]
            else:
                openai_data[param] = options[param]
            logging.debug(f"Parameter {param} set from options: {options[param]}")

@app.route('/status', methods=['GET'])
@require_api_key
def status():
    return jsonify(model_server.get_status())

@app.route('/unload', methods=['GET'])
@require_api_key
def unload():
    model_server.stop_model_process()
    logging.info("Stopping the model process.")
    return jsonify({"status": "success"})

@app.route('/api/show', methods=['GET', 'POST'])
@require_api_key
def api_show_dummy():
    logging.info("Request: /api/show")
    return jsonify({"version": "0.0.1"})

@app.route('/v1/completions', methods=['POST'])
@require_api_key
def completions():
    with model_server.process_lock:
        data = request.json
        is_stream = data.get('stream', True)
        logging.debug(f"Received completions request data: {json.dumps(data, ensure_ascii=False)}")
        success, error_msg = prepare_model_request(data)
        if not success:
            if is_stream:
                def error_gen():
                    yield json.dumps({"error": error_msg}).encode('utf-8') + b"\n\n"
                return Response(stream_with_context(error_gen()), content_type="application/json")
            return jsonify({"error": error_msg})
        template = model_server.model_template or {}
        update_from_template(data, template)
        max_tokens = data.get('max_tokens')
        max_seq_len = model_server.model_params.get('max_seq_len')
        if max_tokens is not None and max_seq_len is not None and max_tokens > max_seq_len:
            logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Removing max_tokens.")
            data.pop('max_tokens', None)
        if is_stream:
            def generate():
                try:
                    with requests.post(f"{API_ENDPOINT}/v1/completions", json=data, stream=True) as resp:
                        for line in resp.iter_lines():
                            if line:
                                yield line + b"\n\n"
                                model_server.last_access_time = time.time()
                                logging.debug(f"Streaming response: {line}")
                        logging.debug("Finished streaming response")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request failed: {e}")
                    yield f"error: {str(e)}\n\n".encode('utf-8')
            return Response(stream_with_context(generate()), content_type="application/json")
        else:
            try:
                with requests.post(f"{API_ENDPOINT}/v1/completions", json=data, stream=False) as resp:
                    try:
                        result = resp.json()
                    except json.JSONDecodeError:
                        return jsonify({"error": "Failed to decode JSON response from model server."})
                model_server.last_access_time = time.time()
                return jsonify(result)
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                return jsonify({"error": str(e)})

@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    with model_server.process_lock:
        data = request.json
        is_stream = data.get('stream', True)
        keep_alive = data.get('keep_alive')
        model_server.unload_timer = keep_alive if keep_alive is not None else model_server.default_unload_timer
        model = data.get('model')
        if model and model != model_server.current_model:
            logging.info(f"Model changed: {model_server.current_model} -> {model}")
            if not model_server.start_model_process(model):
                return jsonify({"error": f"Config file for model {model} not found."})
            model_server.current_model = model
        for _ in range(60):
            if model_server.is_server_ready():
                break
            time.sleep(1)
        if not model_server.server_ready_event.wait(timeout=60):
            return jsonify({"error": "Server is not ready"})
        model_server.last_access_time = time.time()
        max_tokens = data.get('max_tokens')
        max_seq_len = model_server.model_params.get('max_seq_len')
        if max_tokens is not None and max_seq_len is not None and max_tokens > max_seq_len:
            logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Removing max_tokens.")
            data.pop('max_tokens', None)
        data.pop('model', None)
        data.pop('keep_alive', None)
        template = model_server.model_template or {}
        update_from_template(data, template)
        inject_system_message(data, template)
        logging.debug(f"Modified request data: {json.dumps(data, ensure_ascii=False)}")
        if is_stream:
            if DEBUG_OUTPUT:
                reply_txt = ""
            def generate():
                try:
                    with requests.post(f"{API_ENDPOINT}/v1/chat/completions", json=data, stream=True) as resp:
                        if DEBUG_OUTPUT:
                            print("Started streaming /v1/chat/completions response:\n", end="", flush=True)
                        for line in resp.iter_lines():
                            if line:
                                yield line + b"\n\n"
                                model_server.last_access_time = time.time()
                                if DEBUG_OUTPUT:
                                    try:
                                        json_str = line.decode('utf-8').replace('data: ', '')
                                        if json_str == '[DONE]':
                                            logging.debug("Stream finished")
                                        else:
                                            parsed_data = json.loads(json_str)
                                            if 'content' in parsed_data['choices'][0]['delta']:
                                                content = parsed_data['choices'][0]['delta']['content']
                                                #reply_txt += content
                                                print(content, end="", flush=True)
                                    except Exception as e:
                                        logging.debug(f"Can't parse response. {e}")
                                        logging.debug(f"JSON String: {json_str}")
                        print("\n", end="", flush=True)
                        logging.debug("Finished streaming /v1/chat/completions response")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Request failed: {e}")
                    yield f"error: {str(e)}\n\n".encode('utf-8')

            return Response(stream_with_context(generate()), content_type="text/event-stream")
        else:
            try:
                with requests.post(f"{API_ENDPOINT}/v1/chat/completions", json=data, stream=False) as resp:
                    try:
                        result = resp.json()
                    except json.JSONDecodeError:
                        return jsonify({"error": "Failed to decode JSON response from model server."})
                model_server.last_access_time = time.time()
                return jsonify(result)
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                return jsonify({"error": str(e)})

@app.route('/v1/models', methods=['GET'])
@require_api_key
def models():
    model_ids = list_model_ids()
    response_data = {
        "object": "list",
        "data": [{"id": mid, "object": "model", "owned_by": "tabbyAPI"} for mid in model_ids]
    }
    return jsonify(response_data)

# Ollama API emulation endpoints
@app.route('/api/version', methods=['GET'])
@require_api_key
def version():
    return jsonify({"version": "0.4.6"})

@app.route('/api/tags', methods=['GET'])
@require_api_key
def tags():
    model_ids = list_model_ids()
    models_list = []
    for mid in model_ids:
        model_name = f"{mid}:exl2"
        digest = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
        models_list.append({"name": model_name, "model": model_name, "digest": digest})
    return jsonify({"models": models_list})

@app.route('/api/chat', methods=['POST'])
@require_api_key
def ollama_chat():
    def generate():
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received Ollama chat request data: {json.dumps(data, ensure_ascii=False)}")
            keep_alive = data.get('keep_alive')
            model_server.unload_timer = keep_alive if keep_alive is not None else model_server.default_unload_timer
            model = data.get('model')
            ollama_model = model
            if model and ':exl2' in model:
                model = model.split(':')[0]
            if model and model != model_server.current_model:
                logging.info(f"Model changed: {model_server.current_model} -> {model}")
                if not model_server.start_model_process(model):
                    yield json.dumps({"error": f"Config file for model {model} not found."}).encode('utf-8') + b"\n\n"
                    return
                model_server.current_model = model
            for _ in range(60):
                if model_server.is_server_ready():
                    break
                time.sleep(1)
            if not model_server.server_ready_event.wait(timeout=60):
                yield json.dumps({"error": "Server is not ready"}).encode('utf-8') + b"\n\n"
                return
            model_server.last_access_time = time.time()
            template = model_server.model_template or {}
            inject_system_message(data, template)
            logging.debug(f"Modified Ollama chat request data: {json.dumps(data, ensure_ascii=False)}")
            openai_data = {"messages": data.get('messages', []), "stream": data.get('stream', True)}
            update_from_template(openai_data, template)
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
            logging.debug(f"Final OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}")
            try:
                if openai_data.get("stream"):
                    with requests.post(f"{API_ENDPOINT}/v1/chat/completions", json=openai_data, stream=True) as resp:
                        if DEBUG_OUTPUT:
                            print("Started streaming /v1/chat/completions response:\n", end="", flush=True)
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode('utf-8')
                                if line_str.startswith('data: '):
                                    line_str = line_str[5:]
                                if line_str.strip() == '[DONE]':
                                    logging.debug("Received [DONE] signal. Finishing response stream.")
                                    break
                                try:
                                    openai_response = json.loads(line_str)
                                except json.JSONDecodeError:
                                    logging.error(f"Failed to decode JSON: {line_str}")
                                    continue
                                delta = openai_response.get("choices", [{}])[0].get("delta", {})
                                role = delta.get("role", "")
                                content = delta.get("content", "")
                                is_done = openai_response.get("finish_reason", "") == "stop"
                                ollama_response = {
                                    "model": ollama_model,
                                    "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                    "message": {"role": role, "content": content, "images": None},
                                    "done": is_done
                                }
                                yield json.dumps(ollama_response).encode('utf-8') + b"\n"
                                model_server.last_access_time = time.time()
                                if DEBUG_OUTPUT:
                                    try:
                                        print(content, end="", flush=True)
                                    except Exception as e:
                                        logging.debug(f"Can't parse response. {e}")
                                        logging.debug(f"JSON String: {json_str}")
                        else:
                            ollama_response = {
                                "model": ollama_model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "done": True
                            }
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n"
                            logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                else:
                    with requests.post(f"{API_ENDPOINT}/v1/chat/completions", json=openai_data, stream=False) as resp:
                        if resp.status_code == 200:
                            try:
                                openai_response = resp.json()
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {resp.text}")
                                yield json.dumps({"error": "Failed to decode JSON response"}).encode('utf-8') + b"\n\n"
                                return
                            message = openai_response.get("choices", [{}])[0].get("message", {})
                            role = message.get("role", "")
                            content = message.get("content", "")
                            is_done = openai_response.get("finish_reason", "") == "stop"
                        else:
                            role, content, is_done = "", "", False
                        ollama_response = {
                            "model": ollama_model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "message": {"role": role, "content": content, "images": None},
                            "done": is_done
                        }
                        yield json.dumps(ollama_response).encode('utf-8') + b"\n"
                        model_server.last_access_time = time.time()
                        logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n".encode('utf-8')
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream" if request.json.get('stream', True) else "application/json")

@app.route('/api/generate', methods=['POST'])
@require_api_key
def ollama_generate():
    def generate():
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received Ollama generate request data: {json.dumps(data, ensure_ascii=False)}")
            keep_alive = data.get('keep_alive')
            model_server.unload_timer = keep_alive if keep_alive is not None else model_server.default_unload_timer
            model = data.get('model')
            ollama_model = model
            if model and ':exl2' in model:
                model = model.split(':')[0]
            if model and model != model_server.current_model:
                logging.info(f"Model changed: {model_server.current_model} -> {model}")
                if not model_server.start_model_process(model):
                    yield json.dumps({"error": f"Config file for model {model} not found."}).encode('utf-8') + b"\n\n"
                    return
                model_server.current_model = model
            for _ in range(60):
                if model_server.is_server_ready():
                    break
                time.sleep(1)
            if not model_server.server_ready_event.wait(timeout=60):
                yield json.dumps({"error": "Server is not ready"}).encode('utf-8') + b"\n\n"
                return
            model_server.last_access_time = time.time()
            openai_data = {
                "prompt": data.get('prompt', ''),
                "suffix": data.get('suffix', ''),
                "stream": data.get('stream', True)
            }
            template = model_server.model_template or {}
            update_from_template(openai_data, template)
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
            logging.debug(f"Final OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}")
            try:
                if openai_data.get("stream"):
                    with requests.post(f"{API_ENDPOINT}/v1/completions", json=openai_data, stream=True) as resp:
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode('utf-8')
                                if line_str.startswith('data: '):
                                    line_str = line_str[5:]
                                if line_str.strip() == '[DONE]':
                                    logging.debug("Received [DONE] signal. Finishing response stream.")
                                    break
                                try:
                                    openai_response = json.loads(line_str)
                                except json.JSONDecodeError:
                                    logging.error(f"Failed to decode JSON: {line_str}")
                                    continue
                                text = openai_response.get("choices", [{}])[0].get("text", "")
                                is_done = openai_response.get("finish_reason", "") == "stop"
                                if DEBUG_OUTPUT:
                                    print(text, end="", flush=True)
                                ollama_response = {
                                    "model": ollama_model,
                                    "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                    "response": text,
                                    "done": is_done
                                }
                                yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                        else:
                            ollama_response = {
                                "model": ollama_model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "response": "",
                                "done": True
                            }
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                            logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                else:
                    with requests.post(f"{API_ENDPOINT}/v1/completions", json=openai_data, stream=False) as resp:
                        if resp.status_code == 200:
                            try:
                                openai_response = resp.json()
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {resp.text}")
                                yield json.dumps({"error": "Failed to decode JSON response"}).encode('utf-8') + b"\n\n"
                                return
                            text = openai_response.get("choices", [{}])[0].get("text", "")
                            is_done = openai_response.get("finish_reason", "") == "stop"
                            if DEBUG_OUTPUT:
                                print(text, end="", flush=True)
                            ollama_response = {
                                "model": ollama_model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "response": text,
                                "done": is_done
                            }
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                            model_server.last_access_time = time.time()
                            logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                        else:
                            logging.error(f"Request failed with status code: {resp.status_code}")
                            yield json.dumps({"error": f"Request failed with status code: {resp.status_code}"}).encode('utf-8') + b"\n\n"
            except requests.exceptions.RequestException as e:
                logging.error(f"Ollama generate request failed: {e}")
                yield f"error: {str(e)}\n\n".encode('utf-8')
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream" if request.json.get('stream', True) else "application/json")

if __name__ == '__main__':
    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=FLASK_DEBUG)
