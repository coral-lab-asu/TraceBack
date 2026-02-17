import os
import random
import time
from pathlib import Path
from typing import Optional

try:
	import openai  # type: ignore
	from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
	openai = None
	OpenAI = None

try:
	from google import genai  # type: ignore
except Exception:  # pragma: no cover
	genai = None


def _load_dotenv_from_root():
	"""
	Lightweight .env loader (KEY=VALUE per line) at repo root.
	Only sets variables that are not already present in os.environ.
	"""
	try:
		root = Path(__file__).resolve().parents[1]
		env_path = root / ".env"
		if not env_path.exists():
			return
		for raw in env_path.read_text(encoding="utf-8").splitlines():
			line = raw.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key and key not in os.environ:
				os.environ[key] = value
	except Exception:
		return


_load_dotenv_from_root()


def _require_env(name: str) -> str:
	api_key = os.getenv(name)
	if not api_key:
		raise RuntimeError(f"Missing required environment variable: {name}")
	return api_key


class Call_OpenAI:
	def __init__(self, model: str = "gpt-4o"):
		self.count = 0
		self.model = model
		if OpenAI is None:
			raise RuntimeError("openai package is required for Call_OpenAI (pip install openai)")
		api_key = _require_env("OPENAI_API_KEY")
		self.client = OpenAI(api_key=api_key)
		self.total_input_tokens = 0
		self.total_output_tokens = 0
		self.total_tokens = 0

	def __repr__(self) -> str:
		return (
			f"Model: {self.model}, Api Calls: {self.count}, "
			f"Input Tokens: {self.total_input_tokens}, "
			f"Output Tokens: {self.total_output_tokens}, "
			f"Total Tokens: {self.total_tokens}"
		)

	def call(self, prompt: str) -> str:
		try:
			completion = self.client.chat.completions.create(
				model=self.model,
				messages=[
					{
						"role": "developer",
						"content": "You are a helpful, instruction-following assistant.",
					},
					{"role": "user", "content": prompt},
				],
			)
			usage = completion.usage
			if usage is not None:
				self.total_input_tokens += usage.prompt_tokens
				self.total_output_tokens += usage.completion_tokens
				self.total_tokens += usage.total_tokens
				print(
					f"API Call #{self.count + 1} - Input: {usage.prompt_tokens}, "
					f"Output: {usage.completion_tokens}, Total: {usage.total_tokens} tokens"
				)
			self.count += 1
			return completion.choices[0].message.content.strip()
		except Exception as e:
			print(f"An unexpected error occurred: {e}")
			raise


class Call_Gemini:
	def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
		if genai is None:
			raise RuntimeError("google-genai package is required for Call_Gemini (pip install google-genai)")
		keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
		if not keys_env:
			raise RuntimeError("Set GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY for Gemini access")
		self.gemini_api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]

		self.model = model_name
		self.count = 0
		self.total_input_tokens = 0
		self.total_output_tokens = 0
		self.total_tokens = 0

	def __repr__(self) -> str:
		return (
			f"Model: {self.model}, Api Calls: {self.count}, "
			f"Input Tokens: {self.total_input_tokens}, "
			f"Output Tokens: {self.total_output_tokens}, "
			f"Total Tokens: {self.total_tokens}"
		)

	def call(self, prompt: str) -> str:
		n = len(self.gemini_api_keys)
		generation_config = {
			"temperature": 0.01,
			"max_output_tokens": 1024,
			"response_mime_type": "text/plain",
		}
		api_key = self.gemini_api_keys[self.count % n]
		client = genai.Client(api_key=api_key)
		local = 0
		while True:
			try:
				response = client.models.generate_content(
					model=self.model,
					contents=prompt,
					config=generation_config,
				)
				self.count += 1
				break
			except Exception:
				local += 1
				api_key = self.gemini_api_keys[local % n]
				client = genai.Client(api_key=api_key)
				time.sleep(1)

		self.total_input_tokens += response.usage_metadata.prompt_token_count
		self.total_output_tokens += response.usage_metadata.candidates_token_count
		self.total_tokens = self.total_input_tokens + self.total_output_tokens
		print(
			f"API Call #{self.count} - Tokens- Input: {response.usage_metadata.prompt_token_count}, "
			f"Output: {response.usage_metadata.candidates_token_count}"
		)
		return response.text.strip()


class Call_DeepSeek:
	def __init__(self, model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):
		self.count = 0
		self.model = model
		if OpenAI is None or openai is None:
			raise RuntimeError("openai package is required for Call_DeepSeek (pip install openai)")
		api_key = os.getenv("DEEPINFRA_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
		if not api_key:
			raise RuntimeError("Set DEEPINFRA_API_KEY or DEEPSEEK_API_KEY for DeepSeek/DeepInfra access")
		openai.api_base = "https://api.deepinfra.com/v1/openai"
		openai.api_key = api_key
		self.client = OpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=api_key)

	def __repr__(self) -> str:
		return f"Model: {self.model}, Api Calls: {self.count}"

	def call(self, prompt: str) -> str:
		try:
			response = self.client.chat.completions.create(
				model="meta-llama/Llama-3.3-70B-Instruct",
				messages=[
					{
						"role": "system",
						"content": "You are an expert at instruction-following. Just output the final answer, nothing else.",
					},
					{"role": "user", "content": prompt},
				],
				max_tokens=512,
				temperature=0.01,
			)
			self.count += 1
			return response.choices[0].message.content.strip()
		except Exception as e:
			print(e)
			raise

			

class Call_Novita:
	"""
	OpenAI-compatible client for Novita AI.

	Expected env var:
	  - NOVITA_API_KEY

	Example model:
	  - qwen/qwen3-235b-a22b-instruct-2507
	"""

	def __init__(
		self,
		model: str = "qwen/qwen3-235b-a22b-instruct-2507",
		*,
		rpm: Optional[float] = None,
		max_retries: int = 10,
		max_tokens: int = 512,
		temperature: float = 0.0,
		top_p: float = 1.0,
		base_url: str = "https://api.novita.ai/openai",
	):
		self.count = 0
		self.model = model
		self.rpm = float(rpm) if rpm is not None else float(os.getenv("NOVITA_RPM", "10"))
		self.max_retries = int(os.getenv("NOVITA_MAX_RETRIES", str(max_retries)))
		self.max_tokens = int(max_tokens)
		self.temperature = float(temperature)
		self.top_p = float(top_p)
		self._min_interval_sec = 60.0 / self.rpm if self.rpm > 0 else 0.0
		self._next_request_ts = 0.0

		if OpenAI is None:
			raise RuntimeError("openai package is required for Call_Novita (pip install openai)")
		api_key = _require_env("NOVITA_API_KEY")
		self.client = OpenAI(api_key=api_key, base_url=base_url)

		# Qwen3 models recommend non-greedy decoding even in non-thinking mode.
		is_qwen3 = "qwen3" in str(model).lower()
		if is_qwen3:
			if self.temperature == 0.0:
				self.temperature = 0.7
			if self.top_p == 1.0:
				self.top_p = 0.8

	def __repr__(self) -> str:
		return f"Model: {self.model} (novita), Calls: {self.count}"

	def _throttle(self) -> None:
		if self._min_interval_sec <= 0:
			return
		now = time.monotonic()
		wait = self._next_request_ts - now
		if wait > 0:
			time.sleep(wait)
		self._next_request_ts = time.monotonic() + self._min_interval_sec

	def _is_rate_limit(self, err: Exception) -> bool:
		status = getattr(err, "status_code", None) or getattr(err, "http_status", None)
		if status == 429:
			return True
		if openai is not None:
			rle = getattr(openai, "RateLimitError", None)
			if rle is not None and isinstance(err, rle):
				return True
		msg = str(err)
		return ("RATE_LIMIT" in msg) or ("429" in msg)

	def _retry_after_sec(self, err: Exception) -> Optional[float]:
		for attr in ("response", "http_response"):
			resp = getattr(err, attr, None)
			headers = getattr(resp, "headers", None) if resp is not None else None
			if not headers:
				continue
			ra = headers.get("Retry-After") or headers.get("retry-after")
			if not ra:
				continue
			try:
				return float(ra)
			except Exception:
				continue
		headers = getattr(err, "headers", None)
		if headers:
			ra = headers.get("Retry-After") or headers.get("retry-after")
			if ra:
				try:
					return float(ra)
				except Exception:
					return None
		return None

	def call(self, prompt: str) -> str:
		system_prompt = "You are an expert at instruction-following. Just output the final answer, nothing else."
		if "qwen3" in str(self.model).lower():
			system_prompt += " Do not include any hidden reasoning or <think> blocks."

		last_err = None
		for attempt in range(max(1, self.max_retries) + 1):
			self._throttle()
			try:
				resp = self.client.chat.completions.create(
					model=self.model,
					messages=[
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": prompt},
					],
					max_tokens=self.max_tokens,
					temperature=self.temperature,
					top_p=self.top_p,
				)
				self.count += 1
				text = resp.choices[0].message.content.strip()
				if "</think>" in text:
					text = text.rsplit("</think>", 1)[-1].strip()
				return text
			except Exception as e:
				last_err = e
				if not self._is_rate_limit(e):
					raise
				retry_after = self._retry_after_sec(e)
				backoff = min(60.0, (2.0 ** min(attempt, 6)) + random.random())
				sleep_sec = max(self._min_interval_sec, retry_after or 0.0, backoff)
				time.sleep(sleep_sec)
				continue

		if last_err is not None:
			raise last_err
		raise RuntimeError("Novita call failed without an exception")


class Call_HF:
	"""
	Minimal Hugging Face Transformers caller for local (GPU/CPU) models.
	Designed to match the `call(prompt: str) -> str` interface used by TraceBackWorkflowRunner.
	"""

	def __init__(
		self,
		model_id: str,
		*,
		max_new_tokens: int = 512,
		temperature: float = 0.0,
		top_p: float = 1.0,
		top_k: Optional[int] = None,
		min_p: Optional[float] = None,
		cache_dir: Optional[str] = None,
		device_map: str = "auto",
		torch_dtype: str = "auto",
		enable_thinking: Optional[bool] = None,
		system_prompt: str = "You are a helpful, instruction-following assistant. Output only what the prompt asks for.",
	):
		self.count = 0
		self.model_id = model_id
		self.max_new_tokens = int(max_new_tokens)
		self.temperature = float(temperature)
		self.top_p = float(top_p)
		self.top_k = int(top_k) if top_k is not None else None
		self.min_p = float(min_p) if min_p is not None else None
		self.cache_dir = cache_dir
		self.device_map = device_map
		self.torch_dtype = torch_dtype
		# Qwen3 defaults to thinking mode; for this project we want no-think output
		# (parsers expect only the tagged final answer lines).
		is_qwen3 = "qwen3" in str(model_id).lower()
		if enable_thinking is None and is_qwen3:
			enable_thinking = False
		# Qwen3 non-thinking recommended defaults (per model card):
		# Temperature=0.7, TopP=0.8, TopK=20, MinP=0 (avoid greedy decoding).
		if is_qwen3 and enable_thinking is False:
			if self.temperature == 0.0:
				self.temperature = 0.7
			if self.top_p == 1.0:
				self.top_p = 0.8
			if self.top_k is None:
				self.top_k = 20
			if self.min_p is None:
				self.min_p = 0.0
		self.enable_thinking = enable_thinking
		self.system_prompt = system_prompt

		try:
			from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
		except Exception as e:  # pragma: no cover
			raise RuntimeError("Call_HF requires transformers (and torch). Install: pip install transformers accelerate torch") from e

		token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
		auth_kwargs = {"token": token} if token else {}

		self.tok = AutoTokenizer.from_pretrained(
			model_id,
			trust_remote_code=True,
			cache_dir=cache_dir,
			**auth_kwargs,
		)
		try:
			if getattr(self.tok, "pad_token_id", None) is None and getattr(self.tok, "eos_token_id", None) is not None:
				self.tok.pad_token_id = self.tok.eos_token_id
		except Exception:
			pass

		self.model = AutoModelForCausalLM.from_pretrained(
			model_id,
			trust_remote_code=True,
			device_map=device_map,
			torch_dtype=torch_dtype,
			cache_dir=cache_dir,
			**auth_kwargs,
		)
		self.gen = pipeline(
			task="text-generation",
			model=self.model,
			tokenizer=self.tok,
			return_full_text=False,
		)

	def __repr__(self) -> str:
		return f"Model: {self.model_id} (hf), Calls: {self.count}"

	def call(self, prompt: str) -> str:
		prompt_text = prompt
		try:
			if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
				messages = [
					{"role": "system", "content": self.system_prompt},
					{"role": "user", "content": prompt},
				]
				kwargs = {"tokenize": False, "add_generation_prompt": True}
				if self.enable_thinking is not None:
					kwargs["enable_thinking"] = bool(self.enable_thinking)
				try:
					prompt_text = self.tok.apply_chat_template(messages, **kwargs)
				except TypeError:
					kwargs.pop("enable_thinking", None)
					prompt_text = self.tok.apply_chat_template(messages, **kwargs)
		except Exception:
			prompt_text = prompt

		gen_kwargs = {
			"max_new_tokens": self.max_new_tokens,
			"do_sample": self.temperature > 0.0,
			"temperature": self.temperature,
			"top_p": self.top_p,
		}
		if self.top_k is not None:
			gen_kwargs["top_k"] = self.top_k
		if self.min_p is not None:
			gen_kwargs["min_p"] = self.min_p

		try:
			out = self.gen(prompt_text, **gen_kwargs)
		except TypeError:
			# Older transformers may not support min_p; retry without it.
			gen_kwargs.pop("min_p", None)
			out = self.gen(prompt_text, **gen_kwargs)
		if isinstance(out, list) and out:
			text = out[0].get("generated_text", "")
		else:
			text = str(out)
		self.count += 1
		text = str(text).strip()
		# If a reasoning model emits <think>...</think>, drop the thinking block.
		if "</think>" in text:
			text = text.rsplit("</think>", 1)[-1].strip()
		return text
