import os, subprocess, time, threading, json, logging, sys, yaml, hashlib, atexit, requests, queue
from flask import Flask, request, Response, stream_with_context, jsonify, make_response
from flask_cors import CORS
from datetime import datetime, timezone
from functools import wraps

DEBUG_OUTPUT = True
FLASK_DEBUG = True

REQUEST_PRINT = False
RESPONSE_PRINT = False

PROXY_HOST = '127.0.0.1'
PROXY_PORT = 9000
DEFAULT_UNLOAD_TIMER = 900
TABBY_API_ENDPOINT = "http://127.0.0.1:7001"
OLLAMA_API_ENDPOINT = "http://127.0.0.1:11434"
CONFIG_DIR = "config"
MODEL_DIR = "models"
TABBY_CONFIG_PATH = "config.yml"
API_TOKENS_PATH = "api_tokens.yml"
ALLOWED_HOSTS = ['api.externaldomain.com']

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG if DEBUG_OUTPUT else logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

if os.path.exists(TABBY_CONFIG_PATH):
    with open(TABBY_CONFIG_PATH, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    host = main_config.get('network', {}).get('host', '127.0.0.1')
    port = main_config.get('network', {}).get('port', 5000)
    TABBY_API_ENDPOINT = f"http://{host}:{port}"
    MODEL_DIR = main_config.get('model', {}).get('model_dir', 'models')
    logging.debug(f"Started process parsed from {TABBY_CONFIG_PATH}: API endpoint {TABBY_API_ENDPOINT}; model directory: {MODEL_DIR}")
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
        if request.headers.get('Host') in ALLOWED_HOSTS:
            auth_header = request.headers.get('Authorization', '')
            token = auth_header.split('Bearer ')[-1] if auth_header.startswith('Bearer ') else ''
            if not token or token not in api_tokens.values():
                logging.error("Authorization header is missing or invalid.")
                return jsonify({"error": "FORBIDDEN"}), 403
        return func(*args, **kwargs)
    return wrapper

def stream_with_release(generator, release_func):
    try:
        for item in generator:
            yield item
    finally:
        release_func()

class ModelServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelServer, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.default_unload_timer = 900
        self.unload_timer = self.default_unload_timer
        self.lock = threading.Condition()
        self.current_model = None
        self.active_requests = 0
        self.switch_queue = queue.Queue()
        self.server_ready = False
        self.server_ready_event = threading.Event()
        self.last_access_time = None
        self.python_executable = sys.executable
        self.model_params = None
        self.draft_model_params = None
        self.model_template = None
        self.current_source = None
        threading.Thread(target=self.switch_worker, daemon=True).start()
        threading.Thread(target=self.check_last_access_time, daemon=True).start()
        atexit.register(self.stop_model_process)

    def acquire_model_lock(self, data):
        requested_model = data.get('model')
        if not requested_model:
            return False, "Model ID is required"
        model_info = get_model_info(requested_model)
        if not model_info:
            return False, "Model not found"

        with self.lock:
            if self.current_model is None:
                success, err = prepare_model_request(data)
                if not success:
                    return False, err
                self.current_model = requested_model
                self.active_requests += 1
                logging.debug(f"acquire_model_lock: Loaded model '{requested_model}', active_requests={self.active_requests}")
                return True, None
            if self.current_model == requested_model:
                self.active_requests += 1
                logging.debug(f"acquire_model_lock: Model '{requested_model}' already active, active_requests={self.active_requests}")
                return True, None
            switch_event = threading.Event()
            self.switch_queue.put((requested_model, switch_event))
            logging.debug(f"acquire_model_lock: Registered switch request for model '{requested_model}'. Queue size={self.switch_queue.qsize()}")
        
        switch_event.wait()
        with self.lock:
            if self.current_model != requested_model:
                return False, "Model switch did not occur as expected"
            self.active_requests += 1
            logging.debug(f"acquire_model_lock: After switch, model '{requested_model}' active, active_requests={self.active_requests}")
            return True, None

    def release_model_lock(self):
        with self.lock:
            self.active_requests -= 1
            logging.debug(f"release_model_lock: active_requests decreased to {self.active_requests}")
            if self.active_requests == 0:
                self.lock.notify_all()

    def switch_worker(self):
        while True:
            with self.lock:
                if self.active_requests > 0 or self.switch_queue.empty():
                    self.lock.wait(timeout=1)
                    continue
            try:
                target_model, event = self.switch_queue.get_nowait()
            except queue.Empty:
                continue
            with self.lock:
                if self.active_requests > 0:
                    self.switch_queue.put((target_model, event))
                    continue
                dummy_data = {'model': target_model}
                success, err = prepare_model_request(dummy_data)
                if success:
                    old_model = self.current_model
                    self.current_model = target_model
                    logging.info(f"switch_worker: Model switched from '{old_model}' to '{target_model}'")
                    self.lock.notify_all()
                    event.set()
                else:
                    logging.error(f"switch_worker: Failed to switch to '{target_model}': {err}")
                    self.switch_queue.put((target_model, event))
            time.sleep(0.1)

    def stop_model_process(self):
        try:
            if self.current_source == "tabbyAPI":
                self.stop_model_process_tabby()
            elif self.current_source == "ollama":
                self.stop_model_process_ollama(self.current_model)
        except Exception as e:
            logging.error(f"Error stopping model process: {e}")

    def start_model_process_tabby(self, model):
        if hasattr(self, 'current_process') and self.current_process is not None:
            self.stop_model_process()
        config_path = os.path.join(CONFIG_DIR, model.replace('/', os.sep) + '.yml')
        if not os.path.commonprefix([os.path.abspath(config_path), os.path.abspath(CONFIG_DIR)]).startswith(os.path.abspath(CONFIG_DIR)):
            logging.error(f"start_model_process_tabby: Forbidden path: {config_path}")
            return False
        if not os.path.exists(config_path):
            logging.error(f"start_model_process_tabby: Config file for model {model} not found.")
            return False
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        self.model_params = model_config.get('model', {})
        self.draft_model_params = model_config.get('draft_model', {})
        self.model_template = model_config.get('template', {})
        model_name = self.model_params.get('model_name')
        if not model_name:
            logging.error(f"start_model_process_tabby: Model name not found for model {model}.")
            return False
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            logging.error(f"start_model_process_tabby: Model directory not found: {model_path}")
            return False
        self.unload_timer = self.model_params.get('unload_timer', self.default_unload_timer)
        logging.debug(f"start_model_process_tabby: Unload timer set to {self.unload_timer}s.")
        if DEBUG_OUTPUT:
            self.current_process = subprocess.Popen(
                [self.python_executable, "start.py", "--config", config_path]
            )
        else:
            self.current_process = subprocess.Popen(
                [self.python_executable, "start.py", "--config", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        logging.debug(f"start_model_process_tabby: Process for model {model} started with PID {self.current_process.pid}")
        self.server_ready = False
        self.server_ready_event.clear()
        threading.Thread(target=self.check_server_ready, daemon=True).start()
        self.last_access_time = time.time()
        self.current_model = model
        self.current_source = "tabbyAPI"
        return True

    def stop_model_process_tabby(self):
        if self.current_process is not None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("stop_model_process_tabby: Terminate failed. Killing process...")
                self.current_process.kill()
            logging.debug(f"stop_model_process_tabby: Process with PID {self.current_process.pid} stopped")
            self.current_process = None
            self.server_ready = False
            self.server_ready_event.clear()
            self.last_access_time = None
            self.current_model = None
            self.current_source = None
            self.unload_timer = self.default_unload_timer
            self.model_params = None
            self.draft_model_params = None
            self.model_template = None

    def start_model_process_ollama(self, model):
        logging.info(f"start_model_process_ollama: Loading Ollama model: {model}")
        self.current_model = model
        self.current_source = "ollama"
        self.last_access_time = time.time()
        self.unload_timer = self.model_params.get('unload_timer', self.default_unload_timer) if self.model_params else self.default_unload_timer
        return True

    def stop_model_process_ollama(self, model):
        try:
            logging.info(f"stop_model_process_ollama: Unloading Ollama model: {model}")
            response = requests.post(f"{OLLAMA_API_ENDPOINT}/api/chat", json={"model": model, "messages": [], "keep_alive": 0})
            response.raise_for_status()
            for _ in range(30):
                ps_response = requests.get(f"{OLLAMA_API_ENDPOINT}/api/ps")
                ps_response.raise_for_status()
                ps_data = ps_response.json()
                if not ps_data.get('models', []):
                    logging.info(f"stop_model_process_ollama: Model {model} successfully unloaded.")
                    break
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            logging.error(f"stop_model_process_ollama: Failed to unload model {model}: {e}")
            return False
        except Exception as e:
            logging.error(f"stop_model_process_ollama: Unexpected error unloading model {model}: {e}")
            return False
        self.current_model = None
        self.current_source = None
        self.last_access_time = None
        return True

    def is_ollama_model_loaded(self, model):
        try:
            response = requests.get(f"{OLLAMA_API_ENDPOINT}/api/ps")
            response.raise_for_status()
            models = response.json().get('models', [])
            return any(m['name'] == model for m in models)
        except requests.exceptions.RequestException as e:
            logging.error(f"is_ollama_model_loaded: Failed to check model {model}: {e}")
            return False

    def is_server_ready(self, endpoint):
        try:
            url = f"{endpoint}/health" if self.current_source == "tabbyAPI" else f"{endpoint}/api/ps"
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logging.debug(f"is_server_ready: Failed at {endpoint}: {e}")
            return False

    def check_server_ready(self):
        endpoint = TABBY_API_ENDPOINT if self.current_source == "tabbyAPI" else OLLAMA_API_ENDPOINT
        for _ in range(30):
            if self.is_server_ready(endpoint):
                self.server_ready = True
                self.server_ready_event.set()
                logging.debug("check_server_ready: Server is ready.")
                break
            time.sleep(1)
        else:
            logging.error("check_server_ready: Server not ready after 30 seconds. Terminating process.")
            self.stop_model_process()

    def check_last_access_time(self):
        while True:
            if self.last_access_time is not None:
                elapsed = time.time() - self.last_access_time
                if elapsed > self.unload_timer:
                    logging.info(f"check_last_access_time: No requests for {elapsed} seconds. Stopping model process.")
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
            "model_template": self.model_template,
            "current_source": self.current_source
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
    data.pop('model', None)
    
def inject_system_message(data, template):
    system_time_flag = template.get("system_time", False)
    system_time_str = ""
    if system_time_flag:
        now = datetime.now()
        formatted_time = now.strftime('%H:%M:%S %Z%z')
        formatted_date = now.strftime('%Y-%m-%d')
        system_time_str = f"Current time: {formatted_time}\nToday: {formatted_date}"
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

def enforce_max_tokens_limit(data):
    max_tokens = data.get('max_tokens')
    max_seq_len = model_server.model_params.get('max_seq_len') if model_server.model_params else None
    if max_tokens is not None:
        if max_tokens < 4:
            logging.debug(f"max_tokens ({max_tokens}) is less than 4. Removing max_tokens.")
            data.pop('max_tokens', None)
        if max_seq_len is not None and max_tokens > max_seq_len:
            logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Removing max_tokens.")
            data.pop('max_tokens', None)

def list_model_ids():
    tabby_model_ids = []
    for root, _, files in os.walk(CONFIG_DIR):
        for file in files:
            if file.endswith('.yml'):
                rel_path = os.path.relpath(root, CONFIG_DIR)
                model_id = os.path.splitext(file)[0] if rel_path == '.' else os.path.join(rel_path, os.path.splitext(file)[0]).replace(os.sep, '/')
                tabby_model_ids.append({"id": model_id, "source": "tabbyAPI"})
    try:
        response = requests.get(f"{OLLAMA_API_ENDPOINT}/v1/models")
        response.raise_for_status()
        ollama_models = response.json().get('data', [])
        ollama_model_ids = [{"id": model['id'], "source": "ollama"} for model in ollama_models]
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch models from Ollama API: {e}")
        ollama_model_ids = []
    return tabby_model_ids + ollama_model_ids

def get_model_info(model_id):
    return next((m for m in list_model_ids() if m['id'] == model_id), None)

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
    openai_data.pop('model', None)

def stream_error_gen(error_message):
    def generator():
        yield json.dumps({"error": error_message}).encode('utf-8') + b"\n\n"
    return generator

def prepare_model_request(data):
    model = data.get('model')
    if not model:
        return False, "Model parameter is missing."
    model_info = get_model_info(model)
    if not model_info:
        return False, "Model not found."
    source = model_info['source']
    if model != model_server.current_model:
        logging.info(f"Model changed: {model_server.current_model} -> {model}")
        model_server.stop_model_process()
        if source == "tabbyAPI":
            if not model_server.start_model_process_tabby(model):
                return False, f"Config file for model {model} not found."
        elif source == "ollama":
            logging.info(f"Switching to Ollama model: {model_server.current_model} -> {model}")
            if not model_server.start_model_process_ollama(model):
                return False, f"Failed to load model {model} on Ollama."
        else:
            return False, "Unknown model source."
        model_server.current_model = model
        model_server.current_source = source
    for _ in range(60):
        endpoint = TABBY_API_ENDPOINT if model_server.current_source == "tabbyAPI" else OLLAMA_API_ENDPOINT
        if model_server.is_server_ready(endpoint):
            break
        time.sleep(1)
    if source == "tabbyAPI" and not model_server.server_ready_event.wait(timeout=60):
        return False, "Server is not ready"
    if 'keep_alive' in data:
        model_server.unload_timer = data['keep_alive']
    else:
        model_server.unload_timer = model_server.model_params.get('unload_timer', model_server.default_unload_timer) if model_server.model_params else DEFAULT_UNLOAD_TIMER
        if source == "ollama":
            data['keep_alive'] = DEFAULT_UNLOAD_TIMER
    model_server.last_access_time = time.time()
    return True, None


@app.route('/', methods=['GET'])
@require_api_key
def ollama_response():
    response = make_response("Ollama is running")
    response.headers['Content-Type'] = 'text/plain'
    return response

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
    return jsonify({"version": "0.4.6"})

@app.route('/v1/completions', methods=['POST'])
@require_api_key
def completions():
    data = request.json
    logging.debug(f"Received completions request data: {json.dumps(data, ensure_ascii=False)}")
    requested_model = data.get('model')
    is_stream = data.get('stream', True)
    if not requested_model:
        return jsonify({"error": "Model ID is required"}), 400
    model_info = get_model_info(requested_model)
    if not model_info:
        return jsonify({"error": "Model not found"}), 404
    success, error_message = model_server.acquire_model_lock(data)
    if not success:
        if is_stream:
            return Response(stream_with_context(stream_error_gen(error_message)()), content_type="application/json")
        return jsonify({"error": error_message})
    if is_stream:
        def generate():
            try:
                with requests.post(f"{TABBY_API_ENDPOINT if model_info['source']=='tabbyAPI' else OLLAMA_API_ENDPOINT}/v1/completions", json=data, stream=True) as resp:
                    for line in resp.iter_lines():
                        if line:
                            yield line + b"\n\n"
                            model_server.last_access_time = time.time()
                            logging.debug(f"Streaming response: {line}")
                    logging.debug("Finished streaming response")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n".encode('utf-8')
        return Response(stream_with_context(stream_with_release(generate(), model_server.release_model_lock)), content_type="text/event-stream")
    else:
        try:
            resp = requests.post(f"{TABBY_API_ENDPOINT if model_info['source']=='tabbyAPI' else OLLAMA_API_ENDPOINT}/v1/completions", json=data, stream=False)
            try:
                result = resp.json()
            except json.JSONDecodeError:
                return jsonify({"error": "Failed to decode JSON response from model server."})
            model_server.last_access_time = time.time()
            return resp
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return jsonify({"error": str(e)})
        finally:
            model_server.release_model_lock()

@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    data = request.json
    if DEBUG_OUTPUT or REQUEST_PRINT:
        print(f"\nOriginal request:\n{json.dumps(data, indent=4)}\n---------------------------------\n")
    logging.debug(f"Received chat completions request data: {json.dumps(data, ensure_ascii=False)}")
    requested_model = data.get('model')
    is_stream = data.get('stream', True)
    if not requested_model:
        return jsonify({"error": "Model ID is required"}), 400
    model_info = get_model_info(requested_model)
    if not model_info:
        return jsonify({"error": "Model not found"}), 404
    success, error_message = model_server.acquire_model_lock(data)
    if not success:
        if is_stream:
            return Response(stream_with_context(stream_error_gen(error_message)()), content_type="application/json")
        return jsonify({"error": error_message})
    if model_info['source'] == "tabbyAPI":
        api_endpoint = TABBY_API_ENDPOINT
        template = model_server.model_template or {}
        update_from_template(data, template)
        inject_system_message(data, template)
        openai_data = data
    elif model_info['source'] == "ollama":
        api_endpoint = OLLAMA_API_ENDPOINT
        openai_data = data
    else:
        model_server.release_model_lock()
        return jsonify({"error": "Unknown model source"}), 500
    enforce_max_tokens_limit(data)
    if DEBUG_OUTPUT or REQUEST_PRINT:
        print(f"\nModified request:\n{json.dumps(openai_data, ensure_ascii=False, indent=4)}\n---------------------------------\n")
    if is_stream:
        def generate():
            try:
                with requests.post(f"{api_endpoint}/v1/chat/completions", json=openai_data, stream=True) as resp:
                    if DEBUG_OUTPUT or RESPONSE_PRINT:
                        print(f"\n\nResponse:\n\n")
                    for line in resp.iter_lines():
                        if line:
                            yield line + b"\n\n"
                            line_decoded = line.decode('utf-8')
                            if line_decoded.startswith('data: '):
                                line_decoded = line_decoded[5:]
                            if line_decoded.strip() == '[DONE]':
                                logging.debug("Received [DONE] signal. Finishing response stream.")
                                break
                            try:
                                openai_response = json.loads(line_decoded)
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {line_decoded}")
                                continue
                            delta = openai_response.get("choices", [{}])[0].get("delta", {})
                            if openai_response.get("finish_reason", "") == "":
                                is_done = False
                                pass
                            else:
                                print(openai_response, end="", flush=True)
                                is_done = openai_response.get("finish_reason", "") == "stop"
                            if is_done:
                                if DEBUG_OUTPUT:
                                    logging.debug("Stream finished")
                                break
                            if DEBUG_OUTPUT or RESPONSE_PRINT:
                                try:
                                    if 'choices' in openai_response and 'delta' in openai_response['choices'][0] and 'content' in openai_response['choices'][0]['delta']:
                                        content = openai_response['choices'][0]['delta']['content']
                                        print(content, end="", flush=True)
                                except Exception as e:
                                    if DEBUG_OUTPUT:
                                        logging.debug(f"Can't parse response. {e}")
                            model_server.last_access_time = time.time()
                    if DEBUG_OUTPUT or RESPONSE_PRINT:
                        print(f"\n---------------------------------\n")
                    if DEBUG_OUTPUT:
                        logging.info("Finished streaming /v1/chat/completions response")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n".encode('utf-8')
        return Response(stream_with_context(stream_with_release(generate(), model_server.release_model_lock)), content_type="text/event-stream", direct_passthrough=True)
    else:
        try:
            with requests.post(f"{api_endpoint}/v1/chat/completions", json=openai_data, stream=False) as resp:
                try:
                    result = resp.json()
                except json.JSONDecodeError:
                    return jsonify({"error": "Failed to decode JSON response from model server."})
            model_server.last_access_time = time.time()
            return jsonify(result)
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return jsonify({"error": str(e)})
        finally:
            model_server.release_model_lock()

@app.route('/v1/models', methods=['GET'])
@require_api_key
def models():
    model_details = list_model_ids()
    response_data = {
        "object": "list",
        "data": [{"id": mid["id"], "object": "model", "owned_by": mid["source"]} for mid in model_details]
    }
    return jsonify(response_data)

# Ollama API Emulation
@app.route('/api/version', methods=['GET'])
@require_api_key
def version():
    return jsonify({"version": "0.4.6"})

@app.route('/api/tags', methods=['GET'])
@require_api_key
def tags():
    model_details = list_model_ids()
    models_list = []
    for mid in model_details:
        if mid['source'] == "tabbyAPI":
            model_name = f"{mid['id']}:exl2"
            digest = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
            models_list.append({"name": model_name, "model": model_name, "digest": digest})
    try:
        response = requests.get(f"{OLLAMA_API_ENDPOINT}/api/tags")
        response.raise_for_status()
        ollama_models = response.json().get('models', [])
        for ollama_model in ollama_models:
            models_list.append({
                "name": ollama_model["name"],
                "model": ollama_model["model"],
                "digest": ollama_model["digest"]
            })
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch models from Ollama API: {e}")

    return jsonify({"models": models_list})

@app.route('/api/chat', methods=['POST'])
@require_api_key
def ollama_chat():
    def generate():
        data = request.json
        if REQUEST_PRINT:
            print(f"\n---------------------------------\nOriginal /api/chat request:\n\n{data}\n\n---------------------------------\n")
        logging.debug(f"Received Ollama chat request data: {json.dumps(data, ensure_ascii=False)}")
        is_stream = data.get('stream', True)
        data['stream'] = is_stream
        ollama_model = data.get('model')
        if ':exl2' in ollama_model:
            ollama_model = ollama_model.split(':')[0]
        data['model'] = ollama_model
        if not ollama_model:
            logging.debug("Ollama Model ID is required")
            yield json.dumps({"error": "Model ID is required"}).encode('utf-8') + b"\n\n"
            return
        model_info = get_model_info(ollama_model)
        if not model_info:
            logging.debug("Ollama Model not found")
            yield json.dumps({"error": "Model not found"}).encode('utf-8') + b"\n\n"
            return
        source = model_info['source']
        success, error_message = model_server.acquire_model_lock(data)
        if not success:
            yield json.dumps({"error": error_message}).encode('utf-8') + b"\n\n"
            return
        if source == "tabbyAPI":
            api_endpoint = TABBY_API_ENDPOINT
            template = model_server.model_template or {}
            inject_system_message(data, template)
            openai_data = {"messages": data.get('messages', []), "stream": is_stream}
            update_from_template(openai_data, template)
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
        elif source == "ollama":
            api_endpoint = OLLAMA_API_ENDPOINT
            keep_alive = data.get('keep_alive', DEBUG_OUTPUT)
            messages = data.get('messages',[{}])
            openai_data = data
            ollama_messages = []
            for message in messages:
                ollama_message = {
                    'role': message.get('role'),
                }
                content = message.get('content')
                images = message.get('images', None)
                ollama_content = []
                if images:
                    ollama_text = {
                        'type': 'text',
                        'text': content
                    }
                    ollama_content.append(ollama_text)
                    for image in images:
                        ollama_image = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}"
                            }
                        }
                        ollama_content.append(ollama_image)
                    
                    ollama_message['content'] = ollama_content
                else:
                    ollama_message['content'] = content
                ollama_messages.append(ollama_message)
            openai_data['messages'] = ollama_messages
            openai_data['keep_alive'] = keep_alive
        else:
            yield json.dumps({"error": "Unknown model source"}).encode('utf-8') + b"\n\n"
            model_server.release_model_lock()
            return
        if DEBUG_OUTPUT or REQUEST_PRINT:
            print(f"\n---------------------------------\nModified /api/chat request:\n\n{openai_data}\n\n---------------------------------\n")
        try:
            if openai_data.get("stream"):
                with requests.post(f"{api_endpoint}/v1/chat/completions", json=openai_data, stream=True) as resp:
                    if DEBUG_OUTPUT or RESPONSE_PRINT:
                        print("Started streaming /v1/chat/completions response:\n\n", end="", flush=True)
                    for line in resp.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            #print(line)
                            if line.startswith('data: '):
                                line = line[5:]
                            if line.strip() == '[DONE]':
                                logging.debug("Received [DONE] signal. Finishing response stream.")
                                break
                            try:
                                openai_response = json.loads(line)
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {line}")
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
                            if DEBUG_OUTPUT or RESPONSE_PRINT:
                                try:
                                    print(content, end="", flush=True)
                                except Exception as e:
                                    logging.debug(f"Can't print content. {e}")
                    else:
                        ollama_response = {
                            "model": ollama_model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "done": True
                        }
                        yield json.dumps(ollama_response).encode('utf-8') + b"\n"
                        logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                    if DEBUG_OUTPUT or RESPONSE_PRINT:
                        print(f"\n\n---------------------------------\n")
            else:
                resp = requests.post(f"{api_endpoint}/v1/chat/completions", json=openai_data, stream=False)
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
                yield json.dumps({
                    "model": ollama_model,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                    "message": {"role": role, "content": content, "images": None},
                    "done": is_done
                }).encode('utf-8') + b"\n"
                model_server.last_access_time = time.time()
                logging.debug("Ollama response done.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            yield f"error: {str(e)}\n\n".encode('utf-8')
        finally:
            model_server.release_model_lock()
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream" if request.json.get('stream', True) else "application/json",
                    direct_passthrough=True)

@app.route('/api/generate', methods=['POST'])
@require_api_key
def ollama_generate():
    def generate():
        data = request.json
        logging.debug(f"Received Ollama generate request data: {json.dumps(data, ensure_ascii=False)}")
        is_stream = data.get('stream', True)
        ollama_model = data.get('model')
        if ':exl2' in ollama_model:
            ollama_model = ollama_model.split(':')[0]
        data['model'] = ollama_model
        if not ollama_model:
            logging.debug("Ollama Model ID is required")
            yield json.dumps({"error": "Model ID is required"}).encode('utf-8') + b"\n\n"
            return
        model_info = get_model_info(ollama_model)
        if not model_info:
            logging.debug("Ollama Model not found")
            yield json.dumps({"error": "Model not found"}).encode('utf-8') + b"\n\n"
            return
        source = model_info['source']
        success, error_message = model_server.acquire_model_lock(data)
        if not success:
            yield json.dumps({"error": error_message}).encode('utf-8') + b"\n\n"
            return
        if source == "tabbyAPI":
            api_endpoint = TABBY_API_ENDPOINT
            openai_data = {
                "prompt": data.get('prompt', ''),
                "suffix": data.get('suffix', ''),
                "stream": is_stream
            }
            template = model_server.model_template or {}
            update_from_template(openai_data, template)
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
        elif source == "ollama":
            api_endpoint = OLLAMA_API_ENDPOINT
            openai_data = data
        else:
            yield json.dumps({"error": "Unknown model source"}).encode('utf-8') + b"\n\n"
            model_server.release_model_lock()
            return
        logging.debug(f"Final OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}")
        try:
            if openai_data.get("stream"):
                with requests.post(f"{api_endpoint}/v1/completions", json=openai_data, stream=True) as resp:
                    for line in resp.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                line_str = line_str[5:]
                            if line_str.strip() == '[DONE]':
                                logging.debug("Received [DONE] signal. Finishing response stream.")
                                break
                            try:
                                json.loads(line_str)
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {line_str}")
                                continue
                            text = json.loads(line_str).get("choices", [{}])[0].get("text", "")
                            is_done = json.loads(line_str).get("finish_reason", "") == "stop"
                            if DEBUG_OUTPUT:
                                print(text, end="", flush=True)
                            yield json.dumps({
                                "model": ollama_model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "response": text,
                                "done": is_done
                            }).encode('utf-8') + b"\n\n"
                    else:
                        yield json.dumps({
                            "model": ollama_model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "response": "",
                            "done": True
                        }).encode('utf-8') + b"\n\n"
                        logging.debug("Final Ollama response sent.")
            else:
                resp = requests.post(f"{api_endpoint}/v1/completions", json=openai_data, stream=False)
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
                    yield json.dumps({
                        "model": ollama_model,
                        "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                        "response": text,
                        "done": is_done
                    }).encode('utf-8') + b"\n\n"
                    model_server.last_access_time = time.time()
                    logging.debug("Final Ollama response sent.")
                else:
                    logging.error(f"Request failed with status code: {resp.status_code}")
                    yield json.dumps({"error": f"Request failed with status code: {resp.status_code}"}).encode('utf-8') + b"\n\n"
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama generate request failed: {e}")
            yield f"error: {str(e)}\n\n".encode('utf-8')
        finally:
            model_server.release_model_lock()
    return Response(stream_with_context(stream_with_release(generate(), lambda: None)),
                    content_type="text/event-stream" if request.json.get('stream', True) else "application/json",
                    direct_passthrough=True)

@app.route('/api/embeddings', methods=['POST'])
@require_api_key
def ollama_embeddings():
    data = request.json
    logging.debug(f"Received embeddings request data: {json.dumps(data, ensure_ascii=False)}")
    requested_model = data.get('model')
    prompt = data.get('prompt')
    is_stream = data.get('stream', False)
    if not requested_model:
        return jsonify({"error": "Model ID is required"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    model_info = get_model_info(requested_model)
    if not model_info:
        return jsonify({"error": "Model not found"}), 404
    success, error_message = model_server.acquire_model_lock(data)
    if not success:
        if is_stream:
            return Response(stream_with_context(stream_error_gen(error_message)()), content_type="application/json")
        return jsonify({"error": error_message}), 400
    try:
        source = model_info['source']
        if source == "tabbyAPI":
            return jsonify({"error": "Embeddings generation is not supported for this model source"}), 500
        elif source == "ollama":
            api_endpoint = OLLAMA_API_ENDPOINT
            openai_data = {
                "model": requested_model,
                "input": prompt,
                "stream": is_stream
            }
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
        else:
            return jsonify({"error": "Unknown model source"}), 500
        logging.debug(f"Final OpenAI request data for embeddings: {json.dumps(openai_data, ensure_ascii=False)}")
        try:
            if is_stream:
                return jsonify({"error": "Streaming is not supported for embeddings"}), 400
            else:
                resp = requests.post(f"{api_endpoint}/v1/embeddings", json=openai_data, timeout=3600)
                if resp.status_code == 200:
                    try:
                        result = resp.json()
                    except json.JSONDecodeError:
                        return jsonify({"error": "Failed to decode JSON response from model server."})
                    embedding = result.get("data", [{}])[0].get("embedding", [])
                    created_at = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
                    ollama_response = {
                        "embedding": embedding,
                        "created_at": created_at
                    }
                    return jsonify(ollama_response)
                else:
                    logging.error(f"Request failed with status code: {resp.status_code}")
                    return jsonify({"error": f"Request failed with status code: {resp.status_code}"}), resp.status_code
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama embeddings request failed: {e}")
            return jsonify({"error": str(e)}), 500
    finally:
        model_server.release_model_lock()

if __name__ == '__main__':
    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=FLASK_DEBUG)
