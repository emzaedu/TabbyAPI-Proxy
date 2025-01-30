import os, subprocess, time, threading, requests, json, logging, sys, yaml, hashlib, atexit
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
from functools import wraps

debug_output = True
flask_debug = True

proxy_host = '127.0.0.1'
proxy_port = 9000
unload_timer = 86400
api_endpoint = "http://127.0.0.1:7001"
config_dir = "config"
model_dir = 'models'
tabby_config_path = "config.yml"
api_tokens_path = "api_tokens.yml"
api_key_required_hosts = ['api.externaldomain.com']

proxy_running = False
proxy_process = None
unload_timer_thread = None

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG if debug_output else logging.INFO)

if os.path.exists(tabby_config_path):
    with open(tabby_config_path, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    api_endpoint = f"http://{main_config.get('network', {}).get('host', '127.0.0.1')}:{main_config.get('network', {}).get('port', 5000)}"
    model_dir = main_config.get('model', {}).get('model_dir', 'models')
    logging.debug(f"Started process parsed from {tabby_config_path}: api endpoint {api_endpoint}; model dir path: {model_dir}")
else:
    logging.error("Main configuration file not found.")

if os.path.exists(api_tokens_path):
    with open(api_tokens_path, 'r', encoding='utf-8') as f:
        api_tokens = yaml.safe_load(f)
    logging.debug(f"Loaded API tokens from {api_tokens_path}")
else:
    logging.error("API tokens configuration file not found.")
    api_tokens = {}

@app.before_request
def log_request_info():
    logging.debug(f"Request Headers: {request.headers}")
    logging.debug(f"Request Body: {request.get_data(as_text=True)}")
    
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        host_header = request.headers.get('Host')
        if host_header not in allowed_hosts:
            return f(*args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.error("Authorization header is missing or incorrect.")
            return jsonify({"error": "FORBIDDEN"}), 403
        
        token = auth_header.split('Bearer ')[1]
        if token not in api_tokens.values():
            logging.error("Invalid API key.")
            return jsonify({"error": "FORBIDDEN"}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

class ModelServer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelServer, cls).__new__(cls)
            cls._instance.init()
        return cls._instance
    
    def init(self):
        self.default_unload_timer = unload_timer
        self.unload_timer = unload_timer
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
        
        global config_dir, model_dir
        config_path = os.path.join(config_dir, model.replace('/', os.sep) + '.yml')
        
        if not os.path.commonpath([os.path.abspath(config_path), os.path.abspath(config_dir)]) == os.path.abspath(config_dir):
            logging.error(f"Attempted to access a forbidden path: {config_path}")
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
        
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            logging.error(f"Model directory for model {model} not found: {model_path}")
            return False
        
        self.current_process = subprocess.Popen([self.python_executable, "start.py", "--config", config_path])
        logging.debug(f"Started process for model {model} with PID {self.current_process.pid}")
        
        self.server_ready = False
        self.server_ready_event.clear()
        threading.Thread(target=self.check_server_ready, args=()).start()
        self.last_access_time = time.time()
        return True

    def stop_model_process(self):
        if self.current_process is not None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("Terminate failed. Killing...")
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
            response = requests.get(f"{api_endpoint}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def check_last_access_time(self):
        while True:
            if self.last_access_time is not None:
                elapsed_time = time.time() - self.last_access_time
                if elapsed_time > self.unload_timer:
                    logging.info(f"No requests for {elapsed_time} seconds. Stopping the model process.")
                    self.stop_model_process()
            time.sleep(5)

    def get_status(self):
        if self.current_model is None:
            model = None
        else:
            model = self.current_model
        
        if self.last_access_time is None:
            until = None
        else:
            elapsed_time = time.time() - self.last_access_time
            until = max(0, int(self.unload_timer - elapsed_time))
        
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
            "until" : until,
            "until_str": until_str,
            "model_params": self.model_params,
            "draft_model_params": self.draft_model_params,
            "model_template": self.model_template
        }

model_server = ModelServer()

def update_from_template(data, template):
    for param in [
        'top_p', 
        'min_p', 
        'top_k', 
        'temperature', 
        'top_a', 
        'tfs', 
        'frequency_penalty', 
        'mirostat_mode', 
        'mirostat', 
        'mirostat_eta', 
        'mirostat_tau', 
        'repetition_penalty', 
        'presence_penalty', 
        'repetition_range', 
        "min_tokens",
        "generate_window",
        "stop",
        "banned_strings",
        "token_healing",
        "temperature_last",
        "smoothing_factor",
        "skew",
        "xtc_probability",
        "xtc_threshold",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_range",
        "dry_sequence_breakers",
        "dry_sequence_breakers",
        "add_bos_token",
        "ban_eos_token",
        "skip_special_tokens",
        "negative_prompt",
        "json_schema",
        "regex_pattern",
        "grammar_string",
        "speculative_ngram",
        "cfg_scale",
        "max_temp",
        "min_temp",
        "temp_exponent",
        "n",
        "best_of",
        "echo",
        "suffix",
        "user",
        "prompt_template",
        "add_generation_prompt",
        "response_prefix",
        'penalty_range', 
        "stream",
        "banned_tokens",
        "allowed_tokens",
        "logit_bias",
        "stream_options",
        "logprobs",
        "response_format",
        "template_vars",
    ]:
        if param not in data and param in template:
            data[param] = template[param]
            logging.debug(f"Parameter {param} set from template: {data[param]}")

@app.route('/status', methods=['GET'])
@require_api_key
def status():
    status_data = model_server.get_status()
    return jsonify(status_data)

@app.route('/unload', methods=['GET'])
@require_api_key
def unload():
    #with model_server.process_lock:
    model_server.stop_model_process()
    logging.info(f"Stopping the model process.")
    response_data = { "status" : "success" }
    return jsonify(response_data)

@app.route('/api/show', methods=['GET','POST'])
@require_api_key
def api_show_dummy():
    logging.info(f"Request: /api/show")
    response_data = {"version": "0.0.1"}
    return jsonify(response_data)

@app.route('/v1/completions', methods=['POST'])
@require_api_key
def completions():
    def generate():
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received completions request data: {json.dumps(data, ensure_ascii=False)}\n")
            keep_alive = data.get('keep_alive')
            if keep_alive is not None:
                model_server.unload_timer = keep_alive
            else:
                model_server.unload_timer = model_server.default_unload_timer
                
            model = data.get('model')
            if model is not None and model_server.current_model != model:
                logging.info(f"Model changed from {model_server.current_model} to {model}")
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
            update_from_template(data, template)
            
            max_tokens = data.get('max_tokens')
            max_seq_len = model_server.model_params.get('max_seq_len')
            if max_tokens is not None and max_seq_len is not None and max_tokens > max_seq_len:
                logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Setting max_tokens to {max_seq_len}.")
                data.pop('max_tokens', None)
            data.pop('model', None)
            data.pop('keep_alive', None)
            
            try:
                with requests.post(f"{api_endpoint}/v1/completions", json=data, stream=True) as resp:
                    for line in resp.iter_lines():
                        if line:
                            yield line + b"\n\n"
                            model_server.last_access_time = time.time()
                            logging.debug(f"Streaming response: {line}")
                    logging.debug("Finished streaming response")
                logging.debug("\n")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n"

    return Response(stream_with_context(generate()), content_type="application/json")

@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def proxy():
    def generate():
        for header, value in request.headers.items():
            print(f"{header}: {value}")
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received request data: {json.dumps(data, ensure_ascii=False)}\n")
            keep_alive = data.get('keep_alive')
            if keep_alive is not None:
                model_server.unload_timer = keep_alive
            else:
                model_server.unload_timer = model_server.default_unload_timer
                
            model = data.get('model')
            if model is not None and model_server.current_model != model:
                logging.info(f"Model changed from {model_server.current_model} to {model}")
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
            
            max_tokens = data.get('max_tokens')
            max_seq_len = model_server.model_params.get('max_seq_len')
            if max_tokens is not None and max_seq_len is not None and max_tokens > max_seq_len:
                logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Setting max_tokens to {max_seq_len}.")
                data.pop('max_tokens', None)
            
            data.pop('model', None)
            data.pop('keep_alive', None)
            
            template = model_server.model_template or {}
            update_from_template(data, template)
            
            print(model_server.model_template)
            system_time = model_server.model_template.get("system_time", False)
            print(system_time)
            if system_time:
                local_time = datetime.now()
                formatted_time = local_time.strftime('%H:%M:%S %Z%z')
                formatted_date = local_time.strftime('%Y-%m-%d')
                system_time_string = f"Current time:{formatted_time}\nToday: {formatted_date}\n\n"
            
            messages = data.get('messages', [])
            system_instruction = template.get('system')
            if system_instruction and (not messages or messages[0].get('role') != 'system') :
                if system_time:
                    system_instruction = f"{system_time_string} {system_instruction}"
                system_message = {
                    "role": "system",
                    "content": system_instruction
                }
                messages.insert(0, system_message)
            elif messages[0].get('role') == 'system':
                system_instruction = messages[0].get('content')
                if system_time:
                    system_instruction = f"{system_time_string} {system_instruction}"
                system_message = {
                    "role": "system",
                    "content": system_instruction
                }
                messages.insert(0, system_message)
            
            data['messages'] = messages
            logging.debug(f"Modified request data: {json.dumps(data, ensure_ascii=False)}\n")
            
            reply_txt = ""
            try:
                with requests.post(f"{api_endpoint}/v1/chat/completions", json=data, stream=True) as resp:
                    for line in resp.iter_lines():
                        if line:
                            yield line + b"\n\n"
                            model_server.last_access_time = time.time()
                            #logging.debug(f"Streaming response: {line}")
                            if debug_output:
                                try:
                                    json_str = line.decode('utf-8').replace('data: ', '')
                                    if json_str == '[DONE]':
                                        logging.debug("Stream finished")
                                    else:
                                        parsed_data = json.loads(json_str)
                                        content = parsed_data['choices'][0]['delta']['content']
                                        reply_txt += content
                                        print(content,end="",flush=True)
                                except Exception as e:
                                    logging.debug(f"Can't parse response. {e}")
                                    logging.debug(f"JSON String: {json_str}")
                    logging.debug("Finished streaming response")
                    #print(reply_txt)
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n"

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@app.route('/v1/models', methods=['GET'])
@require_api_key
def models():
    global config_dir
    model_files = []
    
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith('.yml'):
                relative_path = os.path.relpath(root, config_dir)
                if relative_path == '.':
                    model_id = os.path.splitext(file)[0]
                else:
                    model_id = os.path.join(relative_path, os.path.splitext(file)[0]).replace(os.sep, '/')
                model_files.append(model_id)
    
    response_data = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "tabbyAPI"
            }
            for model_id in model_files
        ]
    }
    return jsonify(response_data)

# Ollama API emulation

@app.route('/api/version', methods=['GET'])
@require_api_key
def version():
    response_data = {"version": "0.4.6"}
    return jsonify(response_data)

@app.route('/api/tags', methods=['GET'])
@require_api_key
def tags():
    global config_dir
    model_files = []
    
    for root, dirs, files in os.walk(config_dir):
        for file in files:
            if file.endswith('.yml'):
                relative_path = os.path.relpath(root, config_dir)
                if relative_path == '.':
                    model_id = os.path.splitext(file)[0]
                else:
                    model_id = os.path.join(relative_path, os.path.splitext(file)[0]).replace(os.sep, '/')
                model_name = f"{model_id}:exl2"
                model_files.append({
                    "name": model_name,
                    "model": model_name,
                    "digest": hashlib.sha256(model_name.encode('utf-8')).hexdigest()
                })
    
    response_data = {"models": model_files}
    return jsonify(response_data)

def ollama_update_from_options(openai_data, options):
    for param in [
        'stop', 
        'temperature', 
        'mirostat', 
        'mirostat_eta', 
        'mirostat_tau', 
        'top_k', 
        'top_p', 
        'min_p', 
        'frequency_penalty', 
        'tfs_z', 
        "max_tokens",
        #"num_ctx",
        "num_predict"
    ]:
        if param in options:
            if param == 'top_k':
                logging.debug(f"Original value of {param}: {options[param]}")
                openai_data["top_k"] = int(float(options[param]))
                logging.debug(f"Converted value of {param}: {openai_data['top_k']}")
            elif param == 'tfs_z':
                openai_data["tfs"] = options[param]
            elif param == 'num_predict':
                openai_data["max_tokens"] = options[param]
            else:
                openai_data[param] = options[param]
            logging.debug(f"Parameter {param} set from options: {options[param]}")


@app.route('/api/chat', methods=['POST'])
@require_api_key
def ollama_chat():
    def generate():
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received Ollama chat request data: {json.dumps(data, ensure_ascii=False)}\n")
            keep_alive = data.get('keep_alive')
            if keep_alive is not None:
                model_server.unload_timer = keep_alive
            else:
                model_server.unload_timer = model_server.default_unload_timer
            model = data.get('model')
            if model is not None:
                ollama_model = model
                if ':exl2' in model:
                    model = model.split(':')[0]
                if model_server.current_model != model:
                    logging.info(f"Model changed from {model_server.current_model} to {model}")
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
            
            system_time = model_server.model_template.get("system_time", False)
            if system_time:
                local_time = datetime.now()
                formatted_time = local_time.strftime('%H:%M:%S %Z%z')
                formatted_date = local_time.strftime('%Y-%m-%d')
                system_time_string = f"Current time:{formatted_time}\nToday: {formatted_date}\n\n"

            messages = data.get('messages', [])
            system_instruction = model_server.model_template.get('system')
            if system_instruction:
                if not messages or messages[0].get('role') != 'system':
                    if system_time:
                        system_instruction = f"{system_time_string} {system_instruction}"
                    system_message = {
                        "role": "system",
                        "content": system_instruction
                    }
                    messages.insert(0, system_message)
                elif messages[0].get('role') == 'system':
                    system_instruction = messages[0].get('content')
                    if system_time:
                        system_instruction = f"{system_time_string} {system_instruction}"
                    system_message = {
                        "role": "system",
                        "content": system_instruction
                    }
                    messages[0] = system_message
            data['messages'] = messages
            
            template = model_server.model_template or {}
            openai_data = {
                "messages": data.get('messages', []),
                "stream": data.get('stream', True)
            }
            update_from_template(openai_data, template)
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
            logging.debug(f"Final OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}\n")
            
            try:
                with requests.post(f"{api_endpoint}/v1/chat/completions", json=openai_data, stream=True) as resp:
                    for line in resp.iter_lines():
                        if line:
                            if debug_output:
                                try:
                                    json_str = line.decode('utf-8').replace('data: ', '')
                                    if json_str == '[DONE]':
                                        logging.debug("Stream finished")
                                    else:
                                        parsed_data = json.loads(json_str)
                                        content = parsed_data['choices'][0]['delta']['content']
                                        print(content, end="", flush=True)
                                except Exception as e:
                                    logging.debug(f"Can't parse response. {e}")
                                    logging.debug(f"JSON String: {json_str}")
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[5:]
                            if line == ' [DONE]':
                                logging.debug("Received [DONE] signal. Finishing response stream.")
                                break
                            try:
                                openai_response = json.loads(line)
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {line}")
                                continue
                            
                            if "choices" in openai_response and len(openai_response["choices"]) > 0:
                                delta = openai_response["choices"][0].get("delta", {})
                                role = delta.get("role", "")
                                content = delta.get("content", "")
                                is_done = openai_response.get("finish_reason", "") == "stop"
                            else:
                                role = ""
                                content = ""
                                is_done = False

                            ollama_response = {
                                "model": ollama_model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "message": {
                                    "role": role,
                                    "content": content,
                                    "images": None
                                },
                                "done": is_done
                            }
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n"
                            model_server.last_access_time = time.time()
                    else:
                        ollama_response = {
                            "model": ollama_model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "done": True
                        }
                        yield json.dumps(ollama_response).encode('utf-8') + b"\n"
                        logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                logging.debug("\n")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n"

    return Response(stream_with_context(generate()), content_type="text/event-stream")

@app.route('/api/generate', methods=['POST'])
@require_api_key
def ollama_generate():
    def generate():
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received Ollama generate request data: {json.dumps(data, ensure_ascii=False)}\n")
            keep_alive = data.get('keep_alive')
            if keep_alive is not None:
                model_server.unload_timer = keep_alive
            else:
                model_server.unload_timer = model_server.default_unload_timer
            
            model = data.get('model')
            if model is not None:
                ollama_model = model
                if ':exl2' in model:
                    model = model.split(':')[0]
                if model_server.current_model != model:
                    logging.info(f"Model changed from {model_server.current_model} to {model}")
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
                #"model": model,
                "prompt": data.get('prompt', ''),
                "suffix": data.get('suffix', ''),
                "stream": data.get('stream', True)
            }

            template = model_server.model_template or {}
            update_from_template(openai_data, template)
            options = data.get('options', {})
            ollama_update_from_options(openai_data, options)
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
            logging.debug(f"Final OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}\n")
            
            try:
                with requests.post(f"{api_endpoint}/v1/completions", json=openai_data, stream=True) as resp:
                    for line in resp.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[5:]                                    
                            if line == ' [DONE]':
                                logging.debug("Received [DONE] signal. Finishing response stream.")
                                break
                            try:
                                openai_response = json.loads(line)
 
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {line}")
                                continue
                            
                            if "choices" in openai_response and len(openai_response["choices"]) > 0:
                                text = openai_response["choices"][0].get("text", "")
                                is_done = openai_response.get("finish_reason", "") == "stop"
                            else:
                                text = ""
                                is_done = False
                           
                            if debug_output:
                                print(text, end="", flush=True)

                            ollama_response = {
                                "model": ollama_model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "response": text,
                                "done": is_done
                            }
                            
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                            #logging.debug(f"Streaming Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")

                    else:
                        if debug_output:
                            try:
                                if line == '[DONE]':
                                    logging.debug("Stream finished")
                                else:
                                    parsed_data = json.loads(line.strip())
                                    content = parsed_data['choices'][0]['text']
                                    print(content, end="", flush=True)
                            except Exception as e:
                                logging.debug(f"Ollama generate error 2: {e}")
                                logging.debug(f"Ollama generate error 2: {line}")
                                pass
                        ollama_response = {
                            "model": ollama_model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "response": "",
                            "done": True
                        }
                        yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                        #logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                logging.debug("\n")
            except requests.exceptions.RequestException as e:
                logging.error(f"Ollama generate error 3: {e}")
                yield f"error: {str(e)}\n\n"

    return Response(stream_with_context(generate()), content_type="text/event-stream")

if __name__ == '__main__':
    app.run(host=proxy_host, port=proxy_port, debug=flask_debug)
