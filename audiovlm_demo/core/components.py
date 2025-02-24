from __future__ import annotations

import base64
import gc
import io
import os
import time
from pathlib import Path
from typing import Annotated, Any

import httpx
import tomlkit
import torch
from pydantic import AfterValidator
from pydantic_settings import BaseSettings

from audiovlm_demo.core.utils import resolve_path

_ResolvedPath = Annotated[Path, AfterValidator(resolve_path)]


class Config(BaseSettings):
    """
    Class to store configuration for the demo.

    Includes paths to downloaded models, among other thnigs.
    """

    model_path: _ResolvedPath
    aria_model_path: _ResolvedPath
    qwen_audio_model_path: _ResolvedPath

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        path = resolve_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist.")

        with open(path) as file:
            return cls.model_validate(tomlkit.load(file).unwrap())


class AudioVLM:
    molmo_model_id: str = "allenai/Molmo-7B-D-0924"
    aria_model_id: str = "rhymes-ai/Aria"
    qwen_audio_model_id: str = "Qwen/Qwen2-Audio-7B-Instruct"

    runpod_endpoint_url: str = "https://api.runpod.ai/v2/{endpoint_id}/run"
    runpod_status_url: str = (
        "https://api.runpod.ai/v2/{endpoint_id}/status/{request_id}"
    )

    def __init__(self, *, config: Config, model_store: dict | None = None):
        self.config = config
        model_store_keys = {"Loaded", "History", "Model", "Processor"}
        self.api_keys = {}
        self.api_endpoint_ids = {}
        if model_store is not None:
            if not model_store_keys <= model_store.keys():
                raise ValueError(
                    "Argument `model_store` is missing the following "
                    f"keys: {model_store_keys - model_store.keys()}"
                )
            self.model_store = model_store
        else:
            self.model_store = {
                "Loaded": False,
                "History": [],
                "Model": None,
                "Processor": None,
            }

    def model_cleanup(self):
        # global model_info_pane # Placeholder for Panel UI
        if self.model_store["Model"]:
            # Placeholder for Panel UI
            # model_info_pane.object = "<p><b>No Model Loaded</b></p>"
            del self.model_store["Model"]
            del self.model_store["Processor"]
            gc.collect()
            torch.cuda.empty_cache()
            self.model_store["Model"] = None
            self.model_store["Processor"] = None
            self.model_store["Loaded"] = False

    def load_model(self, model_selection: str):
        # Placeholder
        # global # model_info_pane
        if self.model_store["Model"]:
            self.model_cleanup()

        if "runpod" not in self.api_keys:
            self.api_keys["runpod"] = os.environ.get("RUNPOD_API_KEY")

        match model_selection:
            case "Molmo-7B-D-0924":
                if "molmo" not in self.api_endpoint_ids:
                    self.api_endpoint_ids["molmo"] = os.environ.get("MOLMO_ENDPOINT_ID")
                self.model_store["Loaded"] = True
            # case "Molmo-7B-D-0924-4bit":
            #     model_id_or_path = self.molmo_model_id
            #     self.model_store["Processor"] = AutoProcessor.from_pretrained(
            #         model_id_or_path,
            #         trust_remote_code=True,
            #         torch_dtype=torch.bfloat16,
            #         device_map="auto",
            #     )
            #     arguments = {
            #         "device_map": "auto",
            #         "torch_dtype": "auto",
            #         "trust_remote_code": True,
            #     }
            #     quantization_config = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_quant_type="fp4",  # or nf4
            #         bnb_4bit_use_double_quant=False,
            #     )
            #     arguments["quantization_config"] = quantization_config
            #     self.model_store["Model"] = AutoModelForCausalLM.from_pretrained(
            #         model_id_or_path,
            #         **arguments,
            #     )
            #     self.model_store["Loaded"] = True
            case "Aria":
                if "aria" not in self.api_endpoint_ids:
                    self.api_endpoint_ids["aria"] = os.environ.get("ARIA_ENDPOINT_ID")
                self.model_store["Loaded"] = True
            case "Qwen2-Audio":
                if "qwen" not in self.api_endpoint_ids:
                    self.api_endpoint_ids["qwen"] = os.environ.get("QWEN_ENDPOINT_ID")
                self.model_store["Loaded"] = True
            case _:
                pass

    _default_system_prompt = "You are an unbiased, helpful assistant."

    def compile_prompt_gguf(
        self,
        # TODO: Change `Any` to the correct type
        history: list[dict[str, Any]],
        user_name: str,
        assistant_name: str,
        system_prompt: str | None = None,
    ):
        if system_prompt is None:
            system_prompt = self._default_system_prompt

        messages = []

        for i in history:
            if i["role"] == user_name:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"text": i["content"], "type": "text"}],
                    }
                )
            elif i["role"] == assistant_name:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"text": i["content"], "type": "text"}],
                    }
                )
            else:
                pass

        if messages[-1]["role"] == "user":
            messages[-1]["content"].append({"text": None, "type": "image"})
        return messages

    def compile_prompt(
        self,
        # TODO: Change `Any` to the correct type
        history: list[dict[str, Any]],
        user_name: str,
        assistant_name: str,
        system_prompt: str | None = None,
    ):
        if system_prompt is None:
            system_prompt = self._default_system_prompt

        texts = [""]
        for i in history:
            if i["role"] == user_name:
                texts.append(f'<|startoftext|>USER: {i["content"]}\nASSISTANT:')
            elif i["role"] == assistant_name:
                if i["content"][-13:] == "<|endoftext|>":
                    texts.append(f'{i["content"]}\n')
                elif i["content"][-15:] == "<|endoftext|>\n":
                    texts.append(f'{i["content"]}')
                else:
                    texts.append(f'{i["content"]}<|endoftext|>\n')
            else:
                pass
        return "".join(texts)

    def send_receive_requests(self, *, endpoint_id: str, headers: dict, data: dict):
        response = httpx.post(
            self.runpod_endpoint_url.format(endpoint_id=endpoint_id),
            headers=headers,
            json=data,
        )
        response.raise_for_status()

        request_id = response.json()["id"]

        while (
            status_response := httpx.post(
                self.runpod_status_url.format(
                    endpoint_id=endpoint_id,
                    request_id=request_id,
                ),
                headers=headers,
            )
        ).json()["status"] in {"IN_QUEUE", "IN_PROGRESS"}:
            time.sleep(0.1)

        if status_response.json()["status"] == "COMPLETED":
            generated_text = status_response.json()["output"]
        else:
            raise RuntimeError(
                f"The prompt was unsuccessful. The following is the response: {status_response.json()}"
            )
        return generated_text

    # TODO: Add type annotations
    def molmo_callback(self, *, file_name, image, chat_history):
        prompt_full = self.compile_prompt(
            chat_history,
            "User",
            "Assistant",
        )

        with io.BytesIO() as output:
            image.save(
                output,
                format=image.format,
            )
            image = output.getvalue()
        image = base64.b64encode(image).decode("utf8")

        data = {
            "input": {
                "image": image,
                "text": prompt_full,
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['runpod']}",
        }

        generated_text = self.send_receive_requests(
            endpoint_id=self.api_endpoint_ids["molmo"],
            headers=headers,
            data=data,
        )

        return generated_text

    # TODO: Add type annotations
    def aria_callback(self, *, file_name, image, chat_history):
        messages = self.compile_prompt_gguf(
            chat_history,
            "User",
            "Assistant",
        )

        with io.BytesIO() as output:
            image.save(
                output,
                format=image.format,
            )
            image = output.getvalue()
        image = base64.b64encode(image).decode("utf8")

        data = {
            "input": {
                "image": image,
                "messages": messages,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['runpod']}",
        }

        generated_text = self.send_receive_requests(
            endpoint_id=self.api_endpoint_ids["aria"],
            headers=headers,
            data=data,
        )
        return generated_text

    # TODO: Add type annotations
    def qwen_callback(self, *, file_name, audio_file_content, chat_history):
        audio_file_content = base64.b64encode(audio_file_content).decode("utf8")

        messages = [{"role": "system", "content": "You are a helpful assistant."}] + [
            {
                "role": utterance["role"].lower(),
                "content": [
                    {"type": "audio", "audio_url": "filler.wav"},
                    {
                        "type": "text",
                        "text": utterance["content"],
                    },
                ],
            }
            for utterance in chat_history
        ]

        data = {
            "input": {
                "audio_content": audio_file_content,
                "messages": messages,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['runpod']}",
        }

        generated_text = self.send_receive_requests(
            endpoint_id=self.api_endpoint_ids["qwen"],
            headers=headers,
            data=data,
        )
        return generated_text
