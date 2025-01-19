# TabbyAPI Proxy

## Overview
TabbyAPI Proxy is a Flask-based application designed to serve as an intermediary between client requests and the TabbyAPI server. It supports model management, including starting and stopping models based on usage, and provides endpoints that mimic both OpenAI and Ollama APIs. The proxy includes an `unload_timer` feature that allows the model to be unloaded after a specified period of inactivity. The timer duration is set in seconds in the configuration file and can be adjusted by providing a custom `keep_alive` value in the request. If the `keep_alive` parameter is set to a non-zero value (in seconds), it overrides the `unload_timer` to extend the duration before the model is unloaded.

## Installation

### Prerequisites
- Python 3.8 or higher
- Flask and other dependencies installed in the same environment as TabbyAPI

### Steps
1. **Clone the Repository**:
   Clone the repository to your local machine.
   ```bash
   git clone https://github.com/your-repo/tabbyapi-proxy.git
   ```

2. **Copy Contents**:
   Copy all contents from the cloned `tabbyapi-proxy` directory into the same directory where `start.py` from TabbyAPI is located.
   ```bash
   cp -r tabbyapi-proxy/* /path/to/tabbyapi/
   cd /path/to/tabbyapi/
   ```

3. **Install Dependencies**:
   Install the necessary dependencies in the same environment used by TabbyAPI.
   ```bash
   pip install os subprocess time threading requests json logging sys yaml hashlib atexit flask flask-cors
   ```

4. **Run the Proxy**:
   ```bash
   python proxy.py
   ```

## Configuration

- **Configuration Directory (`config/`)**: Contains YAML files for each model configuration.
- **Models Directory (`models/`)**: Contains the actual model files.
- **Paths**:
  - `config_dir`: Directory containing model configuration files. Default is `"config"`.
  - `model_dir`: Directory containing model files. Default is `'models'`.
  - `tabby_config_path`: Path to the main configuration file. Default is `"config.yml"`.
  - `api_tokens_path`: Path to the API tokens configuration file. Default is `"api_tokens.yml"`.

  Some of these paths may be automatically reassigned based on the data in the TabbyAPI configuration file.

### Model Configurations
Model configuration files allow you to specify default parameters for each model. These parameters can be overridden by values provided in the client request. For example, a model configuration file might include default settings for `temperature`, `top_k`, and other parameters.

#### Example Model Configuration
Here is an example of a model configuration file:

```yaml
model:
  model_name: example_model_name
  cache_size: 32768
  max_seq_len: 32768
  cache_mode: Q4
  chunk_size: 1024
sampling:
  override_preset: example_preset
template:
  system_time: True
  system: "You are a helpful and harmless assistant. You are ExampleModel developed by ExampleCorp."
  top_p: 0.8
  top_k: 20
  temperature: 0.7
draft_model:
  draft_model_dir: models
  draft_model_name: example_draft_model_name
  draft_rope_scale:
  draft_rope_alpha: 1.0
  draft_cache_mode: FP16
```

## Endpoints

### `/status`
- **Method**: GET
- **Description**: Returns the current status of the model server.

### `/unload`
- **Method**: GET
- **Description**: Stops the currently running model process.

### `/api/show`
- **Method**: GET, POST
- **Description**: Dummy endpoint showing a static version response.

### `/v1/completions`
- **Method**: POST
- **Description**: Streams completions from the TabbyAPI server.

### `/v1/chat/completions`
- **Method**: POST
- **Description**: Streams chat completions from the TabbyAPI server.

### `/v1/models`
- **Method**: GET
- **Description**: Lists all available models.

### `/api/version`
- **Method**: GET
- **Description**: Returns the version of the Ollama API emulation.

### `/api/tags`
- **Method**: GET
- **Description**: Lists all available models in the Ollama API format.

### `/api/chat`
- **Method**: POST
- **Description**: Streams chat responses emulating the Ollama API.

### `/api/generate`
- **Method**: POST
- **Description**: Streams text generation responses emulating the Ollama API.

## Unload Timer Feature

By default, the `unload_timer` is set to a specific duration (in seconds) in the `proxy.py` file after which the model will be unloaded if no new requests are received. This timer can be adjusted by providing a custom `keep_alive` value in the request. If the `keep_alive` parameter is set to a non-zero value (in seconds), it overrides the `unload_timer` to extend the duration before the model is unloaded.

### Example Request with `keep_alive`
```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_API_KEY" \
-d '{
    "model": "example_model",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "keep_alive": 600  # Override the unload timer to 600 seconds
}'
```

## Usage

### Example Request to `/v1/chat/completions`
```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_API_KEY" \
-d '{
    "model": "example_model",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "keep_alive": 600  # Override the unload timer to 600 seconds
}'
```

### Example Request to `/api/chat`
```bash
curl -X POST http://127.0.0.1:9000/api/chat \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_API_KEY" \
-d '{
    "model": "example_model:exl2",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "options": {
        "temperature": 0.7,
        "top_k": 50,
        "keep_alive": 1200  # Override the unload timer to 1200 seconds
    }
}'
```

## Notes
- Ensure that the `api_tokens.yml` file is correctly configured with valid API keys.
- The `config.yml` file should be present in the same directory as `proxy.py`.
- The `models/` directory should contain the necessary model files referenced in the configuration files.
- The `unload_timer` feature can be adjusted by setting the `keep_alive` parameter in the request. If `keep_alive` is set to a non-zero value (in seconds), it overrides the `unload_timer` to extend the duration before the model is unloaded.
