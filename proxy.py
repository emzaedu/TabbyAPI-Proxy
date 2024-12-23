import os, subprocess, time, threading, requests, json, logging, sys, yaml, hashlib
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
from functools import wraps


proxy_host = '127.0.0.1'
proxy_port = 9000
unload_timer = 300
api_endpoint = "http://127.0.0.1:7001"
config_dir = "config"
model_dir = 'models'
tabby_config_path = "config.yml"
api_tokens_path = "api_tokens.yml"
api_key_required_hosts = ['api.externaldomain.com']

proxy_running = False
proxy_process = None
unload_timer_thread = None

debug_output = True
flask_debug = True
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
    logging.debug(f"Headers: {request.headers}")
    logging.debug(f"Body: {request.get_data(as_text=True)}")
    
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        host_header = request.headers.get('Host')
        if host_header not in api_key_required_hosts:
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
    def __init__(self):
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

    def start_model_process(self, model):
        if self.current_process is not None:
            self.stop_model_process()
        
        global config_dir, model_dir
        config_path = os.path.join(config_dir, f"{model}.yml")
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
        threading.Thread(target=self.check_server_ready, args=()).start()
        self.last_access_time = time.time()
        return True

    def stop_model_process(self):
        if self.current_process is not None:
            self.current_process.terminate()
            self.current_process.wait()
            logging.debug(f"Stopped process with PID {self.current_process.pid}")
            self.current_process = None
            self.server_ready = False
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
            "draft_model_params": self.draft_model_params
        }
            

model_server = ModelServer()

@app.route('/status', methods=['GET'])
@require_api_key
def status():
    status_data = model_server.get_status()
    return jsonify(status_data)

@app.route('/v1/unload', methods=['GET'])
@require_api_key
def unload():
    #with model_server.process_lock:
    model_server.stop_model_process()
    logging.info(f"Stopping the model process.")
    response_data = { "status" : "success" }
    return jsonify(response_data)

@app.route('/v1/completions', methods=['POST'])
@require_api_key
def completions():
    def generate():
        with model_server.process_lock:
            data = request.json
            logging.debug(f"Received completions request data: {json.dumps(data, ensure_ascii=False)}\n")
            
            model = data.get('model')
            if ':latest' in model:
                model = model.split(':')[0]
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
        
            if not model_server.is_server_ready():
                yield json.dumps({"error": "Server is not ready"}).encode('utf-8') + b"\n\n"
                return
            
            model_server.last_access_time = time.time()
            
            max_tokens = data.get('max_tokens')
            max_seq_len = model_server.model_params.get('max_seq_len')
            if max_tokens is not None and max_seq_len is not None and max_tokens > max_seq_len:
                logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Setting max_tokens to {max_seq_len}.")
                #data['max_tokens'] = max_seq_len/2
                data.pop('max_tokens', None)
            data.pop('model', None)
            
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
            
            model = data.get('model')
            if ':latest' in model:
                model = model.split(':')[0]
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
        
            if not model_server.is_server_ready():
                yield json.dumps({"error": "Server is not ready"}).encode('utf-8') + b"\n\n"
                return
            
            model_server.last_access_time = time.time()
            
            max_tokens = data.get('max_tokens')
            max_seq_len = model_server.model_params.get('max_seq_len')
            if max_tokens is not None and max_seq_len is not None and max_tokens > max_seq_len:
                logging.debug(f"max_tokens ({max_tokens}) is greater than max_seq_len ({max_seq_len}). Setting max_tokens to {max_seq_len}.")
                #data['max_tokens'] = max_seq_len/2
                data.pop('max_tokens', None)
            
            data.pop('model', None)
                
            template = model_server.model_template or {}
            for param in ['top_p', 'top_k', 'temperature']:
                if param not in data and param in template:
                    data[param] = template[param]
            
            messages = data.get('messages', [])
            system_instruction = template.get('system')
            if system_instruction and (not messages or messages[0].get('role') != 'system'):
                system_message = {
                    "role": "system",
                    "content": system_instruction
                }
                messages.insert(0, system_message)
            data['messages'] = messages
            
            logging.debug(f"Modified request data: {json.dumps(data, ensure_ascii=False)}\n")
            
            try:
                with requests.post(f"{api_endpoint}/v1/chat/completions", json=data, stream=True) as resp:
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

    return Response(stream_with_context(generate()), content_type="text/event-stream")
@app.route('/v1/models', methods=['GET'])
@require_api_key
def models():
    global config_dir
    model_files = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
    model_ids = [os.path.splitext(f)[0] for f in model_files]
    
    response_data = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "tabbyAPI"
            }
            for model_id in model_ids
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
    model_files = [f for f in os.listdir(config_dir) if f.endswith('.yml')]
    model_ids = [os.path.splitext(f)[0] for f in model_files]
    
    models = []
    for model_id in model_ids:
        model_name = f"{model_id}:latest"
        models.append({
            "name": model_name,
            "model": model_name,
            "digest": hashlib.sha256(model_name.encode('utf-8')).hexdigest()
        })
    
    response_data = {"models": models}
    return jsonify(response_data)

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
                if ':latest' not in model:
                    model = f"{model}:latest"
                if model_server.current_model != model:
                    logging.info(f"Model changed from {model_server.current_model} to {model}")
                    if not model_server.start_model_process(model.split(':')[0]):
                        yield json.dumps({"error": f"Config file for model {model} not found."}).encode('utf-8') + b"\n\n"
                        return
                    model_server.current_model = model
            
            for _ in range(60):
                if model_server.is_server_ready():
                    break
                time.sleep(1)
        
            if not model_server.is_server_ready():
                yield json.dumps({"error": "Server is not ready"}).encode('utf-8') + b"\n\n"
                return
            
            model_server.last_access_time = time.time()
            
            openai_data = {
                #"model": model.split(':')[0],
                "messages": data.get('messages', []),
                "stream": data.get('stream', True)
            }
            options = data.get('options', {})
            openai_data.update({
                #"max_tokens": options.get('num_ctx'),
                #"max_tokens": options.get('num_predict')
                #"min_tokens": options.get('min_tokens'),
                "temperature": options.get('temperature',0.7),
                "top_k": options.get('top_k',20),
                "top_p": options.get('top_p',0.8),
                "min_p": options.get('min_p'),
                "top_a": options.get('top_a'),
                "tfs": options.get('tfs_z'),
                "token_frequency_penalty": options.get('frequency_penalty'),
                "repeat_last_n": options.get('repeat_last_n'),
                "mirostat": options.get('mirostat'),
                "mirostat_eta": options.get('mirostat_eta'),
                "mirostat_tau": options.get('mirostat_tau'),
                "token_repetition_penalty": options.get('repeat_penalty')

            })
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
            
            logging.debug(f"Sending OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}\n")
            
            try:
                with requests.post(f"{api_endpoint}/v1/chat/completions", json=openai_data, stream=True) as resp:
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
                                delta = openai_response["choices"][0].get("delta", {})
                                role = delta.get("role", "")
                                content = delta.get("content", "")
                                is_done = openai_response.get("finish_reason", "") == "stop"
                            else:
                                role = ""
                                content = ""
                                is_done = False

                            ollama_response = {
                                "model": model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "message": {
                                    "role": role,
                                    "content": content,
                                    "images": None
                                },
                                "done": is_done
                            }
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                            model_server.last_access_time = time.time()
                            logging.debug(f"Streaming Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                    else:
                        ollama_response = {
                            "model": model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "done": True
                        }
                        yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
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
                if ':latest' not in model:
                    model = f"{model}:latest"
                if model_server.current_model != model:
                    logging.info(f"Model changed from {model_server.current_model} to {model}")
                    if not model_server.start_model_process(model.split(':')[0]):
                        yield json.dumps({"error": f"Config file for model {model} not found."}).encode('utf-8') + b"\n\n"
                        return
                    model_server.current_model = model
            
            for _ in range(60):
                if model_server.is_server_ready():
                    break
                time.sleep(1)
        
            if not model_server.is_server_ready():
                yield json.dumps({"error": "Server is not ready"}).encode('utf-8') + b"\n\n"
                return
            
            model_server.last_access_time = time.time()
            
            openai_data = {
                #"model": model.split(':')[0],
                "prompt": data.get('prompt', ''),
                "suffix": data.get('suffix', ''),
                "stream": data.get('stream', True)
            }
            options = data.get('options', {})
            openai_data.update({
                "temperature": options.get('temperature', 0.7),
                "top_k": options.get('top_k', 20),
                "top_p": options.get('top_p', 0.8),
                "min_p": options.get('min_p'),
                "top_a": options.get('top_a'),
                "tfs": options.get('tfs_z'),
                "token_frequency_penalty": options.get('frequency_penalty'),
                "repeat_last_n": options.get('repeat_last_n'),
                "mirostat": options.get('mirostat'),
                "mirostat_eta": options.get('mirostat_eta'),
                "mirostat_tau": options.get('mirostat_tau'),
                "token_repetition_penalty": options.get('repeat_penalty')
            })
            openai_data = {k: v for k, v in openai_data.items() if v is not None}
            
            logging.debug(f"Sending OpenAI request data: {json.dumps(openai_data, ensure_ascii=False)}\n")
            
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

                            ollama_response = {
                                "model": model,
                                "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                                "response": text,
                                "done": is_done
                            }
                            yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                            model_server.last_access_time = time.time()
                            logging.debug(f"Streaming Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                    else:
                        ollama_response = {
                            "model": model,
                            "created_at": datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                            "response": "",
                            "done": True
                        }
                        yield json.dumps(ollama_response).encode('utf-8') + b"\n\n"
                        logging.debug(f"Final Ollama response: {json.dumps(ollama_response, ensure_ascii=False)}")
                logging.debug("\n")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                yield f"error: {str(e)}\n\n"

    return Response(stream_with_context(generate()), content_type="text/event-stream")

if __name__ == '__main__':
    app.run(host=proxy_host, port=proxy_port, debug=flask_debug)
