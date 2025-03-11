import json
import os
import random
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import pandas as pd
import time
import torch
from transformers import set_seed
from vllm import LLM
from vllm.sampling_params import SamplingParams

from enums import DocType, LLMTask, ModelSource, PlanType
from utils import file_to_data_url

random.seed(369)
set_seed(369)

# model_path = os.getenv('MODEL_PATH')
# model_name = os.getenv('MODEL_NAME')

summary_mapping_json_path = os.getenv('MAPPING_JSON_PATH')
package_path = '/'.join(os.path.dirname(__file__).split('/'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.set_float32_matmul_precision('high')


class LLMModule:

    def __init__(self, input_dir: str, output_dir: str, plan_type: PlanType, doc_type: DocType, llm_task: LLMTask,
                 source: ModelSource):
        self.max_concurrent = 8
        self.points_inches_constant = 72
        self.plan_type = plan_type.value
        self.doc_type = doc_type.value
        self.llm_task = llm_task
        self.system_prompt = llm_task.value
        self.model_source = source.value
        self.model = None
        self.sampling_params = None
        # self.model_path = model_path
        with open(f"{package_path}/model_config_v2.json") as f:
            model_config_json = json.loads(f.read())
        self.model_config = model_config_json[self.llm_task.name]
        self.model_name = self.model_config["model_name"]

        self.output_dir = output_dir
        self.input_dir = input_dir
        self.document_json_path = f"{package_path}/data_jsons/document_json/{self.plan_type}/{self.doc_type}"
        self.result_dir = f"{self.output_dir}/summary_results/{self.plan_type}/{self.doc_type}/{self.llm_task.name}"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.required_cols = []
        self.mapping_dict_path = ""
        if self.llm_task == LLMTask.EXTRACT_TEXT:
            self.required_cols = ["sub_section", "question", "prompt"]
            self.mapping_dict_path = f"{summary_mapping_json_path}/{self.plan_type}/{self.doc_type}/extract_text_mapping.csv"
        elif self.llm_task == LLMTask.DETAILED_SUMMARY:
            self.required_cols = ["sub_section", "prompt"]
            self.mapping_dict_path = f"{summary_mapping_json_path}/{self.plan_type}/{self.doc_type}/detailed_summary_mapping.csv"
        elif self.llm_task == LLMTask.DESIGN_GRID_CURRENT_PROVISION_SUMMARY or self.llm_task == LLMTask.DESIGN_GRID_FINAL_SUMMARY or self.llm_task == LLMTask.RECONSTRUCT_FROM_SUMMARY:
            self.required_cols = ["design_grid_section", "design_grid_column", "sub_section", "doc_type",
                                  "page_no_in_grid",
                                  "current_provision_summary_prompt", "final_design_summary_prompt",
                                  "reconstruction_question_prompt"]
            self.mapping_dict_path = f"{summary_mapping_json_path}/{self.plan_type}/{self.doc_type}/design_grid_summary_mapping.csv"
        else:
            raise ValueError(f"Unknown LLMTask: {self.llm_task.name}")
        if not os.path.exists(self.mapping_dict_path):
            raise ValueError(f"File {self.mapping_dict_path} does not exist!!!")

        self.current_provision_summary_col = ""
        self.current_provision_changes_col = ""
        self.final_design_summary_col = ""
        if self.plan_type == PlanType.TRANSFER_PLAN.value:
            self.current_provision_summary_col = "column_2"
            self.current_provision_changes_col = "column_4"
            self.final_design_summary_col = "column_5"
            self.confidence_score_col = "column_6"
        elif self.plan_type == PlanType.START_UP_PLAN.value:
            self.current_provision_summary_col = "column_2"
            self.current_provision_changes_col = "column_3"
            self.final_design_summary_col = "column_4"
            self.confidence_score_col = "column_5"
        else:
            raise ValueError(f"Unknown PlanType: {self.plan_type}")

        self.sections_per_page = {}
        self.sub_sections_per_page = {}
        self.questions_per_page = {}
        self.mapping_dict = pd.DataFrame()
        self.model_input = {}
        self.model_output = {}
        self.result_json = {}
        self.final_output = {}

    def generate_model_input_for_extract_text(self):
        start_time = time.time()

        with open(f"{self.input_dir}/questions_per_page.json", "r") as f:
            self.questions_per_page = json.loads(f.read())
        with open(f"{self.input_dir}/sub_sections_per_page.json", "r") as f:
            self.sub_sections_per_page = json.loads(f.read())
        with open(f"{self.input_dir}/sections_per_page.json", "r") as f:
            self.sections_per_page = json.loads(f.read())

        self.mapping_dict = pd.read_csv(self.mapping_dict_path)
        print(f"mapping_dict: {self.mapping_dict.shape}")
        self.mapping_dict = self.mapping_dict[self.required_cols]
        self.mapping_dict.dropna(subset=['sub_section', 'question'], inplace=True)
        self.mapping_dict = self.mapping_dict.drop_duplicates(subset=['sub_section', 'question'], ignore_index=True)
        print(f"mapping_dict: {self.mapping_dict.shape}")

        section_json = {}
        question_json = {}

        for page_no, section_dict in self.sections_per_page.items():
            for idx, fields_dict in enumerate(section_dict):
                section_name = fields_dict["keyword"]
                if fields_dict['bbox'] and section_name not in section_json:
                    section_json[section_name] = fields_dict['bbox']

        for page_no, section_dict in self.questions_per_page.items():
            for section_name, sub_section_dict in section_dict.items():
                if section_name not in question_json:
                    question_json[section_name] = {}
                for sub_section_name, questions_dict in sub_section_dict.items():
                    if sub_section_name not in question_json[section_name]:
                        question_json[section_name][sub_section_name] = {}
                    for question_dict in questions_dict:
                        question_name = question_dict["keyword"]
                        if question_dict['bbox'] and question_name not in question_json[section_name][sub_section_name]:
                            question_json[section_name][sub_section_name][question_name] = {}
                            question_json[section_name][sub_section_name][question_name]["is_table"] = question_dict[
                                "is_table"]

        for page_no, section_dict in self.sub_sections_per_page.items():
            self.model_input[page_no] = {}
            for section_name, sub_section_dict in section_dict.items():
                if sub_section_dict:
                    section_idx = section_name.rsplit('.', 1)[0].split(' ')[1]
                    for idx, fields_dict in enumerate(sub_section_dict):
                        sub_section_name = fields_dict["keyword"]
                        updated_sub_section_name = sub_section_name
                        if "SECTION" in section_name:
                            updated_sub_section_name = f"{section_idx} - {sub_section_name}"
                        else:
                            updated_sub_section_name = f"{section_name} - {sub_section_name}"
                        if fields_dict['bbox'] and updated_sub_section_name not in self.model_input[page_no]:
                            self.model_input[page_no][updated_sub_section_name] = {}
                            if fields_dict["is_table"]:
                                self.model_input[page_no][updated_sub_section_name]["coordinates"] = fields_dict["bbox"]
                                self.model_input[page_no][updated_sub_section_name]["questions"] = {}
                                for question_name, question_dict in question_json[section_name][
                                    sub_section_name].items():
                                    self.model_input[page_no][updated_sub_section_name]["questions"][question_name] = {}
                                    self.model_input[page_no][updated_sub_section_name]["questions"][question_name][
                                        "image_path"] = f"{self.input_dir}/processed_questions/{section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}/{sub_section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}/{question_name.replace(' ', '_').replace('/', '_').replace('-', '_')}.png"
                                    if question_dict["is_table"]:
                                        # TODO:
                                        # prompt_value = self.system_prompt["tabular_text"]
                                        # self.model_input[page_no][updated_sub_section_name]["questions"][question_name]["prompt"] = prompt_value
                                        if not self.mapping_dict.loc[
                                            self.mapping_dict['question'] == question_name].empty:
                                            prompt_value = self.mapping_dict.loc[
                                                self.mapping_dict['question'] == question_name, "prompt"].iloc[0]
                                            self.model_input[page_no][updated_sub_section_name]["questions"][
                                                question_name]["prompt"] = prompt_value
                                    else:
                                        # TODO:
                                        # prompt_value = self.system_prompt["simple_text"]
                                        # self.model_input[page_no][updated_sub_section_name]["questions"][question_name]["prompt"] = prompt_value
                                        if not self.mapping_dict.loc[
                                            self.mapping_dict['question'] == question_name].empty:
                                            prompt_value = self.mapping_dict.loc[
                                                self.mapping_dict['question'] == question_name, "prompt"].iloc[0]
                                            self.model_input[page_no][updated_sub_section_name]["questions"][
                                                question_name]["prompt"] = prompt_value
                            else:
                                # TODO:
                                # prompt_value = self.system_prompt["simple_text"]
                                # self.model_input[page_no][updated_sub_section_name]["prompt"] = prompt_value
                                # self.model_input[page_no][updated_sub_section_name]["image_path"] = f"{self.input_dir}/processed_sub_sections/{section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}/{sub_section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}.png"
                                # self.model_input[page_no][updated_sub_section_name]["coordinates"] = fields_dict["bbox"]

                                if not self.mapping_dict.loc[
                                    self.mapping_dict['sub_section'] == sub_section_name].empty:
                                    prompt_value = self.mapping_dict.loc[
                                        self.mapping_dict['sub_section'] == sub_section_name, "prompt"].iloc[0]
                                    self.model_input[page_no][updated_sub_section_name]["prompt"] = prompt_value
                                    self.model_input[page_no][updated_sub_section_name][
                                        "image_path"] = f"{self.input_dir}/processed_sub_sections/{section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}/{sub_section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}.png"
                                    self.model_input[page_no][updated_sub_section_name]["coordinates"] = fields_dict[
                                        "bbox"]
                                else:
                                    print(f"{sub_section_name} not in mapping dictionary!!!")
                                    # raise ValueError(f"{sub_section_name} not in mapping dictionary!!!")
                else:
                    updated_section_name = section_name
                    if "SECTION" in section_name:
                        section_idx = section_name.rsplit('.', 1)[0].split(' ')[1]
                        section_actual_name = section_name.rsplit('. ', 1)[1]
                        updated_section_name = f"{section_idx} - {section_actual_name}"

                    # TODO:
                    # if updated_section_name not in self.model_input[page_no]:
                    #     self.model_input[page_no][updated_section_name] = {}
                    # prompt_value = self.system_prompt["simple_text"]
                    # self.model_input[page_no][updated_section_name]["prompt"] = prompt_value
                    # self.model_input[page_no][updated_section_name]["image_path"] = f"{self.input_dir}/processed_sections/{section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}.png"
                    # self.model_input[page_no][updated_section_name]["coordinates"] = section_json[section_name]

                    if not self.mapping_dict.loc[self.mapping_dict['sub_section'] == section_name].empty:
                        if updated_section_name not in self.model_input[page_no]:
                            self.model_input[page_no][updated_section_name] = {}
                        prompt_value = \
                            self.mapping_dict.loc[self.mapping_dict['sub_section'] == section_name, "prompt"].iloc[0]
                        self.model_input[page_no][updated_section_name]["prompt"] = prompt_value
                        self.model_input[page_no][updated_section_name][
                            "image_path"] = f"{self.input_dir}/processed_sections/{section_name.replace(' ', '_').replace('/', '_').replace('-', '_')}.png"
                        self.model_input[page_no][updated_section_name]["coordinates"] = section_json[section_name]
                    else:
                        print(f"{section_name} not in mapping dictionary!!!")
                        # raise ValueError(f"{section_name} not in mapping dictionary!!!")

        print(json.dumps(self.model_input))
        print("\n")
        print(f"Time Taken Model input generation: {time.time() - start_time} seconds")

    def generate_model_input_for_detailed_summary(self):
        start_time = time.time()

        with open(f"{self.input_dir}/final_output.json", "r") as f:
            extract_text_output = json.loads(f.read())["extract_text_result"]

        self.mapping_dict = pd.read_csv(self.mapping_dict_path)
        print(f"mapping_dict: {self.mapping_dict.shape}")
        self.mapping_dict = self.mapping_dict[self.required_cols]
        self.mapping_dict.dropna(subset=['sub_section'], inplace=True)
        self.mapping_dict = self.mapping_dict.drop_duplicates(subset=['sub_section'], ignore_index=True)
        print(f"mapping_dict: {self.mapping_dict.shape}")

        section_json = {}

        for page_no, section_dict in self.sections_per_page.items():
            for idx, fields_dict in enumerate(section_dict):
                section_name = fields_dict["keyword"]
                if fields_dict['bbox'] and section_name not in self.model_input:
                    section_json[section_name] = fields_dict['bbox']

        for page_no, sub_section_dict in extract_text_output.items():
            self.model_input[page_no] = {}
            for sub_section_name, sub_section_attr in sub_section_dict.items():
                # if "coordinates" in sub_section_attr:
                self.model_input[page_no][sub_section_name] = {}
                context = sub_section_attr["value"]
                # TODO:
                # # prompt_value = self.system_prompt.replace("{sub_section_name}", sub_section_name).replace("{context}", context)
                prompt_value = self.system_prompt
                if not self.mapping_dict.loc[self.mapping_dict['sub_section'] == sub_section_name].empty:
                    prompt_value = \
                        self.mapping_dict.loc[self.mapping_dict['sub_section'] == sub_section_name, "prompt"].iloc[
                            0]
                else:
                    print(f"{sub_section_name} not in mapping dictionary!!!")
                    # raise ValueError(f"{sub_section_name} not in mapping dictionary!!!")
                self.model_input[page_no][sub_section_name]["prompt"] = prompt_value
                self.model_input[page_no][sub_section_name]["context"] = sub_section_attr["value"]
                self.model_input[page_no][sub_section_name]["coordinates"] = sub_section_attr["coordinates"]

        print(json.dumps(self.model_input))
        print("\n")
        print(f"Time Taken Model input generation: {time.time() - start_time} seconds")

    def generate_model_input_for_design_grid_current_provision_summary(self):
        start_time = time.time()

        with open(f"{self.input_dir}/final_output.json", "r") as f:
            detailed_summary_result_json = json.loads(f.read())

        self.mapping_dict = pd.read_csv(self.mapping_dict_path)
        print(f"mapping_dict: {self.mapping_dict.shape}")
        self.mapping_dict = self.mapping_dict[self.required_cols]
        mapping_dict = self.mapping_dict[self.mapping_dict["doc_type"] == "AA"].copy()
        mapping_dict.dropna(subset=['design_grid_section', 'design_grid_column', 'sub_section'], inplace=True)
        mapping_dict = mapping_dict.drop_duplicates(subset=['design_grid_section', 'design_grid_column', 'sub_section'],
                                                    ignore_index=True)
        print(f"mapping_dict: {mapping_dict.shape}")

        detailed_summary_json = {}
        for fields_dict in detailed_summary_result_json["document_summary_result"]:
            page_no = fields_dict["pageNo"]
            for field_attr in fields_dict["Fields"]:
                field_name = field_attr["key"]
                field_value = field_attr["valueSet"][0]["value"]
                if field_name not in detailed_summary_json:
                    detailed_summary_json[field_name] = {}
                detailed_summary_json[field_name]["value"] = field_value
                detailed_summary_json[field_name]["page_no"] = page_no

        for row in mapping_dict.to_dict(orient='records'):
            page_no_in_grid = row['page_no_in_grid']
            design_grid_section = row['design_grid_section']
            design_grid_column = row['design_grid_column']
            sub_section = row['sub_section']
            prompt = row['current_provision_summary_prompt']
            doc_type = row['doc_type']
            if page_no_in_grid not in self.model_input:
                self.model_input[page_no_in_grid] = {}
            if design_grid_column not in self.model_input[page_no_in_grid]:
                self.model_input[page_no_in_grid][design_grid_column] = {"doc_type": doc_type, "section": [],
                                                                         "sub_section": [],
                                                                         "detailed_summary": [], "page_no_in_aa": [],
                                                                         "prompt": prompt}
            self.model_input[page_no_in_grid][design_grid_column]["section"].append(design_grid_section)
            self.model_input[page_no_in_grid][design_grid_column]["sub_section"].append(sub_section)
            if sub_section in detailed_summary_json:
                self.model_input[page_no_in_grid][design_grid_column]["detailed_summary"].append(
                    detailed_summary_json[sub_section]["value"])
                self.model_input[page_no_in_grid][design_grid_column]["page_no_in_aa"].append(
                    detailed_summary_json[sub_section]["page_no"])
            else:
                # raise ValueError(f"{sub_section} not in detailed_summary_mapping_json!!!")
                print(f"{sub_section} not in detailed_summary_mapping_json!!!")

        print(json.dumps(self.model_input))
        print("\n")
        print(f"Time Taken Model input generation: {time.time() - start_time} seconds")

    def generate_model_input_for_reconstruct_from_summary(self):
        start_time = time.time()

        with open(f"{self.input_dir}/planDesignGrid.json", "r") as f:
            form_json = json.loads(f.read())

        if not os.path.exists(f"{self.document_json_path}/fields.json"):
            raise ValueError(f"Path: fields.json does not exist in path {self.document_json_path}!!!")
        with open(f"{self.document_json_path}/fields.json", 'r') as f:
            data_json = json.loads(f.read())

        section_summary_json = {}
        for page_idx, page_dict in enumerate(form_json["pages"]):
            section_name = page_dict["section_name"]
            if section_name not in section_summary_json:
                section_summary_json[section_name] = {}
            section_summary = ""
            for form_idx, form_element_dict in enumerate(page_dict["form_elements"]):
                if form_element_dict["type"] == "matrix" and "value" in form_element_dict:
                    for value_idx, value_dict in enumerate(form_element_dict["value"]):
                        if value_dict["column_1"]:
                            sub_section_name = value_dict["column_1"].strip().replace("\n\n\n", " ").replace("\n\n",
                                                                                                             " ").replace(
                                "\n", " ")
                            section_summary += f"{sub_section_name}: \n"

                            current_provision_summary = value_dict[self.current_provision_summary_col]
                            final_design_summary = value_dict[self.final_design_summary_col]

                            section_summary += f"\t{final_design_summary} \n\n"
            section_summary_json[section_name] = section_summary

        for idx, field_json in enumerate(data_json):
            title = field_json['title']
            section_name, sub_field_name = title.split(" : ")
            if section_name not in self.model_input:
                self.model_input[section_name] = {}
            if sub_field_name not in self.model_input[section_name]:
                self.model_input[section_name][sub_field_name] = {}
            for ques_json in field_json['fields']:
                ques_field = ques_json['var']
                ques_desc = ques_json['desc']
                final_design_summary = ""
                if section_name in section_summary_json:
                    final_design_summary = section_summary_json[section_name]
                else:
                    print(f"{section_name} not in PlanDesignGrid")
                self.model_input[section_name][sub_field_name][ques_field] = {}
                self.model_input[section_name][sub_field_name][ques_field]['desc'] = ques_desc
                self.model_input[section_name][sub_field_name][ques_field][
                    'final_design_summary'] = final_design_summary

        print(json.dumps(self.model_input))
        print("\n")
        print(f"Time Taken Model input generation: {time.time() - start_time} seconds")

    def invoke_model(self, prompt_messages: List[Dict]) -> Dict[str, Any]:
        """Make a single call to vLLM server"""
        from datetime import datetime

        # Log start time
        start_time = datetime.now()
        start_time_str = start_time.strftime("%H:%M:%S")

        # Calculate prompt tokens
        total_prompt_tokens = 0
        prompt_texts = []
        for msg in prompt_messages:
            if isinstance(msg['content'], list):
                for content in msg['content']:
                    if content['type'] == 'text':
                        total_prompt_tokens += len(content['text'].split())
                        prompt_texts.append(f"{msg['role']}: {content['text']}")
            else:
                total_prompt_tokens += len(msg['content'].split())
                prompt_texts.append(f"{msg['role']}: {msg['content']}")

        payload = {
            "model": self.model_name,
            "messages": prompt_messages,
            "min_tokens": self.model_config["min_tokens"],
            "max_tokens": self.model_config["max_tokens"],
            "temperature": self.model_config["temperature"],
            "top_p": self.model_config["top_p"],
            "top_k": self.model_config["top_k"],
            "stop": self.model_config["stop"],
            "skip_special_tokens": self.model_config["skip_special_tokens"],
            "spaces_between_special_tokens": self.model_config["spaces_between_special_tokens"],
            'seed': self.model_config["seed"],  # Fixed random seed
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(self.model_config["server_url"], json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Calculate completion metrics
        completion_text = result["choices"][0]["message"]["content"].strip()
        completion_tokens = len(completion_text.split())
        total_tokens = total_prompt_tokens + completion_tokens

        # Calculate timing
        end_time = datetime.now()
        end_time_str = end_time.strftime("%H:%M:%S")
        duration = end_time - start_time

        if response.status_code == 200:
            total_tokens = result["usage"]["total_tokens"]
            total_prompt_tokens = result["usage"]["prompt_tokens"]
            completion_tokens = result["usage"]["completion_tokens"]

        # Log all metrics in a single line
        metrics = [
            f"START={start_time_str}",
            f"END={end_time_str}",
            f"DUR={duration.total_seconds():.2f}s",
            f"STATUS={response.status_code}",
            f"PROMPT_TOKENS={total_prompt_tokens}",
            f"COMPL_TOKENS={completion_tokens}",
            f"TOTAL_TOKENS={total_tokens}"
        ]
        print(f"REQUEST_METRICS | {' | '.join(metrics)}")

        return completion_text

    def process_batch_prompts(self, prompts: List[List[Dict]]) -> List[str]:
        """Process a batch of prompts concurrently using ThreadPoolExecutor"""
        from datetime import datetime

        batch_start = datetime.now()
        batch_start_str = batch_start.strftime("%H:%M:%S")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [executor.submit(self.invoke_model, prompt) for prompt in prompts]

            for i, future in enumerate(futures, 1):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    error_msg = f"ERROR | TIME={current_time} | PROMPT={i}/{len(prompts)} | MSG={str(e)}"
                    print(error_msg)
                    results.append(f"Error: {e}")

        # Log batch metrics in a single line
        batch_end = datetime.now()
        batch_end_str = batch_end.strftime("%H:%M:%S")
        batch_duration = batch_end - batch_start
        avg_time = batch_duration.total_seconds() / len(prompts)

        metrics = [
            f"START={batch_start_str}",
            f"END={batch_end_str}",
            f"TOTAL_DUR={batch_duration.total_seconds():.2f}s",
            f"AVG_DUR={avg_time:.2f}s",
            f"PROMPTS={len(prompts)}",
            f"SUCCESS={len([r for r in results if not str(r).startswith('Error:')])}"
        ]
        print(f"BATCH_METRICS | {' | '.join(metrics)}")

        return results

    def process_prompts(self, prompts: List[List[Dict]]) -> List[str]:
        """Process prompts synchronously using ThreadPoolExecutor"""
        return self.process_batch_prompts(prompts=prompts)

    def run_model_for_extract_text(self):
        def format_prompt_for_extract_text(system_prompt: str, user_prompt: str, image_path: str):
            image_source = file_to_data_url(file_path=image_path)
            messages = [
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": user_prompt},
                #     ]
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_source}},
                    ]
                },
            ]
            return messages

        start_time = time.time()
        print(f"================ Running Model =================")

        prompt_messages = []
        for page_no, fields_dict in self.model_input.items():
            for field_name, fields_attr in fields_dict.items():
                if "questions" in fields_attr:
                    for question_name, question_dict in fields_attr["questions"].items():
                        message = format_prompt_for_extract_text(system_prompt=self.system_prompt,
                                                                 user_prompt=question_dict["prompt"],
                                                                 image_path=question_dict["image_path"])
                        prompt_messages.append(message)
                else:
                    message = format_prompt_for_extract_text(system_prompt=self.system_prompt,
                                                             user_prompt=fields_attr["prompt"],
                                                             image_path=fields_attr["image_path"])
                    prompt_messages.append(message)

        results = self.process_prompts(prompts=prompt_messages)

        final_response = ""
        result_idx_counter = 0
        for page_no, fields_dict in self.model_input.items():
            print(f"================ PageNo: {page_no} =================")
            if page_no not in self.result_json:
                self.result_json[page_no] = {}
            message_input = []
            response_final = ""
            for field_name, fields_attr in fields_dict.items():
                if "questions" in fields_attr:
                    if field_name not in self.result_json:
                        self.result_json[page_no][field_name] = {}
                    self.result_json[page_no][field_name]["coordinates"] = fields_attr["coordinates"]
                    for question_name, question_dict in fields_attr["questions"].items():
                        message_input.append(
                            {"image_path": question_dict["image_path"], "user_prompt": question_dict["prompt"]})
                else:
                    if field_name not in self.result_json:
                        self.result_json[page_no][field_name] = {}
                    # if "coordinates" in fields_attr:
                    self.result_json[page_no][field_name]["coordinates"] = fields_attr["coordinates"]
                    message_input.append(
                        {"image_path": fields_attr["image_path"], "user_prompt": fields_attr["prompt"]})
                for msg_input in message_input:
                    user_prompt = msg_input["user_prompt"]
                    print("\n")
                    print("---------")
                    print("Question:")
                    print("---------")
                    print(user_prompt)
                    print("---------")
                    response = results[result_idx_counter]
                    result_idx_counter += 1
                    final_response = final_response + "---------\nQuestion:\n---------\n" + user_prompt + "\n---------\nResponse:\n---------\n" + response + "\n---------\n"
                    print("Response:")
                    print("---------")
                    print(response)
                    # response = self.format_model_response_for_extract_text(input_text=response)
                    print("---------")
                    print("Response formatted:")
                    print("---------")
                    print(response)
                    print("---------")
                    final_response = final_response + "\n---------\nResponse formatted:\n---------\n" + response + "\n---------\n" + "\n\n\n\n----Separator----\n\n\n\n"
                    if not response_final:
                        response_final = response
                    else:
                        response_final = response_final + "\n\n" + response

                print("\n")
                self.result_json[page_no][field_name]["value"] = response_final
                self.result_json[page_no][field_name]["confidence_score"] = 100
                message_input = []
                response_final = ""

        with open(f"{self.result_dir}/final_response.txt", "w") as f:
            f.write(final_response)

        self.final_output["extract_text_result"] = self.result_json
        print(f"Time Taken Model run: {time.time() - start_time} seconds")

    def run_model_for_detailed_summary(self):
        def format_prompt_for_detailed_summary(system_prompt: str, user_prompt: str):
            messages = [
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": system_prompt},
                #     ]
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                    ]
                },
            ]
            return messages

        start_time = time.time()
        print(f"================ Running Model =================")

        prompt_messages = []
        for page_no, fields_dict in self.model_input.items():
            for field_name, fields_attr in fields_dict.items():
                context = fields_attr["context"]
                user_prompt = fields_attr["prompt"]
                user_prompt = user_prompt.replace("{sub_section_name}", field_name).replace("{context}", context)
                message = format_prompt_for_detailed_summary(system_prompt=self.system_prompt, user_prompt=user_prompt)
                prompt_messages.append(message)

        results = self.process_prompts(prompts=prompt_messages)

        final_response = ""
        result_idx_counter = 0
        for page_no, fields_dict in self.model_input.items():
            print(f"================ PageNo: {page_no} =================")
            if page_no not in self.result_json:
                self.result_json[page_no] = {}
            for field_name, fields_attr in fields_dict.items():
                if field_name not in self.result_json:
                    self.result_json[page_no][field_name] = {}
                self.result_json[page_no][field_name]["coordinates"] = fields_attr["coordinates"]
                context = fields_attr["context"]
                user_prompt = fields_attr["prompt"]
                user_prompt = user_prompt.replace("{sub_section_name}", field_name).replace("{context}", context)
                print("\n")
                print("---------")
                print("Question:")
                print("---------")
                print(user_prompt)
                print("---------")
                response = results[result_idx_counter]
                result_idx_counter += 1
                final_response = final_response + "---------\nQuestion:\n---------\n" + user_prompt + "\n---------\nResponse:\n---------\n" + response + "\n---------\n"
                print("Response:")
                print("---------")
                print(response)
                # response = self.format_model_response_for_detailed_summary(input_text=response)
                print("---------")
                print("Response formatted:")
                print("---------")
                print(response)
                print("---------")
                final_response = final_response + "\n---------\nResponse formatted:\n---------\n" + response + "\n---------\n" + "\n\n\n\n----Separator----\n\n\n\n"
                print("\n")
                self.result_json[page_no][field_name]["value"] = response

        with open(f"{self.result_dir}/final_response.txt", "w") as f:
            f.write(final_response)
        print(f"Time Taken Model run: {time.time() - start_time} seconds")

    def run_model_for_design_grid_current_provision_summary(self):
        def format_prompt_for_design_grid_current_provision_summary(system_prompt: str, user_prompt: str, context: str):
            messages = [
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": system_prompt},
                #     ]
                # },
                # {
                #     "role": "user",
                #     "content": [
                #         {"type": "text", "text": f"{user_prompt}\n\n{context}"},
                #     ]
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompt}"},
                    ]
                }
            ]
            return messages

        start_time = time.time()
        print(f"================ Running Model =================")

        prompt_messages = []
        for page_no, fields_dict in self.model_input.items():
            for field_name, fields_attr in fields_dict.items():
                if fields_attr["detailed_summary"]:
                    if len(fields_attr['sub_section']) > 1:
                        context = "\n\n".join(fields_attr["detailed_summary"])
                        user_prompt = fields_attr["prompt"]
                        user_prompt = user_prompt.replace("{context}", context)
                        message = format_prompt_for_design_grid_current_provision_summary(
                            system_prompt=self.system_prompt,
                            user_prompt=user_prompt, context=context)
                        prompt_messages.append(message)

        results = self.process_prompts(prompts=prompt_messages)

        final_response = ""
        result_idx_counter = 0
        for page_no, fields_dict in self.model_input.items():
            print(f"================ PageNo: {page_no} =================")
            if page_no not in self.result_json:
                self.result_json[page_no] = {}
            for field_name, fields_attr in fields_dict.items():
                if field_name not in self.result_json:
                    self.result_json[page_no][field_name] = {}
                self.result_json[page_no][field_name]["section"] = fields_attr['section']
                self.result_json[page_no][field_name][
                    "section_page_no"] = f"{fields_attr['doc_type']} - {fields_attr['page_no_in_aa']}"
                self.result_json[page_no][field_name]["sub_section"] = fields_attr['sub_section']
                response = ""
                user_prompt = ""
                if fields_attr["detailed_summary"]:
                    if len(fields_attr['sub_section']) > 1:
                        context = "\n\n".join(fields_attr["detailed_summary"])
                        user_prompt = fields_attr["prompt"]
                        print("\n")
                        print("---------")
                        print("Question:")
                        print("---------")
                        user_prompt = user_prompt.replace("{context}", context)
                        print(f"{user_prompt}")
                        print("---------")
                        response = results[result_idx_counter]
                        result_idx_counter += 1
                    else:
                        context = fields_attr["detailed_summary"][0]
                        user_prompt = fields_attr["prompt"]
                        print("\n")
                        print("---------")
                        print("Question:")
                        print("---------")
                        print(f"{user_prompt}")
                        print("---------")
                        response = context
                final_response = final_response + "---------\nQuestion:\n---------\n" + user_prompt + "\n---------\nResponse:\n---------\n" + response + "\n---------\n"
                print("Response:")
                print("---------")
                print(response)
                response = self.format_model_response_for_design_grid_current_provision_summary(input_text=response)
                print("---------")
                print("Response formatted:")
                print("---------")
                print(response)
                print("---------")
                final_response = final_response + "\n---------\nResponse formatted:\n---------\n" + response + "\n---------\n" + "\n\n\n\n----Separator----\n\n\n\n"

                print("\n")
                self.result_json[page_no][field_name]["value"] = response

        with open(f"{self.result_dir}/final_response.txt", "w") as f:
            f.write(final_response)
        print(f"Time Taken Model run: {time.time() - start_time} seconds")

    def run_model_for_design_grid_final_summary(self):
        def format_prompt_for_design_grid_final_summary(system_prompt: str, user_prompt: str):
            messages = [
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": system_prompt},
                #     ]
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompt}"},
                    ]
                }
            ]
            return messages

        start_time = time.time()
        print(f"================ Running Model =================")

        with open(f"{self.input_dir}/planDesignGrid.json", "r") as f:
            form_json = json.loads(f.read())

        self.mapping_dict = pd.read_csv(self.mapping_dict_path)
        print(f"mapping_dict: {self.mapping_dict.shape}")
        self.mapping_dict = self.mapping_dict[self.required_cols]
        # mapping_dict = self.mapping_dict[self.mapping_dict["doc_type"] == "AA"].copy()
        mapping_dict = self.mapping_dict.copy()
        # mapping_dict.dropna(subset=['sub_section'], inplace=True)
        # mapping_dict = mapping_dict.drop_duplicates(subset=['design_grid_column', 'sub_section'], ignore_index=True)
        mapping_dict.dropna(subset=['design_grid_section', 'sub_section'], inplace=True)
        mapping_dict = mapping_dict.drop_duplicates(subset=['design_grid_section', 'sub_section'],
                                                    ignore_index=True)
        print(f"mapping_dict: {mapping_dict.shape}")

        self.result_json = form_json

        prompt_messages = []
        for page_idx, page_dict in enumerate(self.result_json["pages"]):
            for form_idx, form_element_dict in enumerate(page_dict["form_elements"]):
                if form_element_dict["type"] == "matrix" and "value" in form_element_dict:
                    for value_idx, value_dict in enumerate(form_element_dict["value"]):
                        if value_dict["column_1"]:
                            sub_section = value_dict["column_1"].strip()
                            current_provision_summary = value_dict[self.current_provision_summary_col]
                            current_provision_changes = value_dict[self.current_provision_changes_col]

                            # if self.plan_type == PlanType.START_UP_PLAN.value:
                            #     # Extract text after "AT Default:"
                            #     match = re.search(r"AT Default:\s*(.*)", current_provision_summary, re.IGNORECASE)
                            #     if match:
                            #         at_default_text = match.group(1).strip()
                            #         print("Extracted text:", at_default_text)
                            #         current_provision_summary = at_default_text
                            #     else:
                            #         print("No 'AT Default' text found.")

                            if not mapping_dict.loc[mapping_dict['sub_section'] == sub_section].empty:
                                user_prompt = mapping_dict.loc[
                                    mapping_dict['sub_section'] == sub_section, "final_design_summary_prompt"].iloc[0]
                                user_prompt = user_prompt.replace("{sub_section}", sub_section).replace(
                                    "{current_provision_summary}", current_provision_summary).replace(
                                    "{current_provision_changes}", current_provision_changes)
                                if current_provision_changes and current_provision_changes != "No Change":
                                    message = format_prompt_for_design_grid_final_summary(
                                        system_prompt=self.system_prompt, user_prompt=user_prompt)
                                    prompt_messages.append(message)
                            else:
                                print(f"{sub_section} not in mapping dictionary!!!")

        results = self.process_prompts(prompts=prompt_messages)

        final_response = ""
        result_idx_counter = 0
        for page_idx, page_dict in enumerate(self.result_json["pages"]):
            print(f"================ PageNo: {page_idx} =================")
            for form_idx, form_element_dict in enumerate(page_dict["form_elements"]):
                if form_element_dict["type"] == "matrix" and "value" in form_element_dict:
                    for value_idx, value_dict in enumerate(form_element_dict["value"]):
                        if value_dict["column_1"]:
                            sub_section = value_dict["column_1"].strip()
                            current_provision_summary = value_dict[self.current_provision_summary_col]
                            current_provision_changes = value_dict[self.current_provision_changes_col]
                            final_design_summary = current_provision_summary

                            # if self.plan_type == PlanType.START_UP_PLAN.value:
                            #     # Extract text after "AT Default:"
                            #     match = re.search(r"AT Default:\s*(.*)", current_provision_summary, re.IGNORECASE)
                            #     if match:
                            #         at_default_text = match.group(1).strip()
                            #         print("Extracted text:", at_default_text)
                            #         current_provision_summary = at_default_text
                            #     else:
                            #         print("No 'AT Default' text found.")

                            if self.plan_type == PlanType.TRANSFER_PLAN.value and not current_provision_changes:
                                self.result_json["pages"][page_idx]["form_elements"][form_idx]["value"][value_idx][
                                    self.current_provision_changes_col] = "No Change"
                            if not mapping_dict.loc[mapping_dict['sub_section'] == sub_section].empty:
                                user_prompt = mapping_dict.loc[
                                    mapping_dict['sub_section'] == sub_section, "final_design_summary_prompt"].iloc[0]
                                print("\n")
                                print("---------")
                                print("Question:")
                                print("---------")
                                user_prompt = user_prompt.replace("{sub_section}", sub_section).replace(
                                    "{current_provision_summary}", current_provision_summary).replace(
                                    "{current_provision_changes}", current_provision_changes)
                                print(f"{user_prompt}")
                                print("---------")

                                if current_provision_changes and current_provision_changes != "No Change":
                                    response = results[result_idx_counter]
                                    result_idx_counter += 1
                                    print("Response:")
                                    print("---------")
                                    print(response)
                                    print("---------")
                                    response = self.format_model_response_for_design_grid_final_summary(
                                        input_text=response)
                                    final_design_summary = response

                                self.result_json["pages"][page_idx]["form_elements"][form_idx]["value"][value_idx][
                                    self.final_design_summary_col] = final_design_summary
                                self.result_json["pages"][page_idx]["form_elements"][form_idx]["value"][value_idx][
                                    self.confidence_score_col] = 100

                                final_response = final_response + "---------\nQuestion:\n---------\n" + user_prompt + "\n---------\nResponse:\n---------\n" + final_design_summary + "\n---------\n"
                                print("Response formatted:")
                                print("---------")
                                print(final_design_summary)
                                print("---------")
                                final_response = final_response + "\n---------\nResponse formatted:\n---------\n" + final_design_summary + "\n---------\n" + "\n\n\n\n----Separator----\n\n\n\n"

                                print("\n")
                            else:
                                print(f"{sub_section} not in mapping dictionary!!!")

        with open(f"{self.result_dir}/final_response.txt", "w") as f:
            f.write(final_response)
        print(f"Time Taken Model run: {time.time() - start_time} seconds")

    def run_model_for_reconstruct_from_summary(self):
        def format_prompt_for_reconstruct_from_summary(system_prompt: str, user_prompt: str):
            messages = [
                # {
                #     "role": "system",
                #     "content": [
                #         {"type": "text", "text": system_prompt},
                #     ]
                # },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompt}"},
                    ]
                }
            ]
            return messages

        start_time = time.time()
        print(f"================ Running Model =================")

        with open(f"{self.input_dir}/planDesignSpec.json", "r") as f:
            survey_studio_question_json = json.loads(f.read())

        self.mapping_dict = pd.read_csv(self.mapping_dict_path)
        print(f"mapping_dict: {self.mapping_dict.shape}")
        self.mapping_dict = self.mapping_dict[self.required_cols]
        mapping_dict = self.mapping_dict.copy()
        mapping_dict.dropna(subset=['design_grid_section', 'sub_section'], inplace=True)
        mapping_dict = mapping_dict.drop_duplicates(subset=['design_grid_section', 'sub_section'],
                                                    ignore_index=True)
        print(f"mapping_dict: {mapping_dict.shape}")

        prompt_messages = []
        for page_idx, page_dict in enumerate(survey_studio_question_json["pages"]):
            section_name = page_dict["section_name"]
            section_questions = ""
            final_design_summary = ""
            for form_idx, form_element_dict in enumerate(page_dict["form_elements"]):
                format = ""
                if form_element_dict["type"] == "h3" and section_questions:
                    if not mapping_dict.loc[mapping_dict['design_grid_section'] == section_name].empty:
                        user_prompt = mapping_dict.loc[
                            mapping_dict['design_grid_section'] == section_name, "reconstruction_question_prompt"].iloc[
                            0]
                        user_prompt = user_prompt.replace("{sub_section}", section_name).replace(
                            "{final_design_summary}", final_design_summary).replace("{question}", section_questions)
                        message = format_prompt_for_reconstruct_from_summary(system_prompt=self.system_prompt,
                                                                             user_prompt=user_prompt)
                        prompt_messages.append(message)
                    else:
                        print(f"{section_name} not in mapping dictionary!!!")

                if form_element_dict["type"] == "h3":
                    sub_section_name = form_element_dict["content"]
                    if section_questions:
                        section_questions = ""
                        # TODO: uncomment if sub_section_name name required
                    # section_questions += f"{sub_section_name}:\n"
                if form_element_dict["type"] in ["group", "text", "textarea", "number", "date", "radiogroup", "select"]:
                    if form_element_dict["type"] == "group":
                        for children in form_element_dict["children"]:
                            question = children["label"].strip()  # .split(". ")[1]
                            question_field = children["ftw_var_name"].strip()
                            question_field_attr = self.model_input[section_name][sub_section_name.split('. ')[1]][
                                question_field]
                            description = question_field_attr['desc']
                            if section_name not in self.model_input:
                                print(f"{section_name} not in Final Design Summary!!!")
                            final_design_summary = question_field_attr['final_design_summary']
                            possible_values = []
                            default_value = ""
                            if "valueFormat" in children:
                                format = children["valueFormat"]
                            if children["type"] in ["radiogroup", "select"]:
                                possible_values = children["dataDictFieldUI"]["possibleValues"]
                                default_value = children["dataDictFieldUI"]["defaultValue"]
                                if possible_values:
                                    possible_values = {val_dict['optionLabel']: val_dict['fieldName'] for val_dict in
                                                       possible_values}
                                    possible_values = list(possible_values.keys())
                                else:
                                    possible_values = []

                            # question_field += ":\n"
                            # question_field += f"\t\tQuestion: {question} ?\n"
                            # # question_field += f"\t\tDescription: {description}\n"
                            # question_field += f"\t\tPossible Values: {possible_values}\n"
                            # question_field += f"\t\tDefault Value: {default_value}\n"
                            #
                            # section_questions += f"\t{question_field}\n"

                            question_field += ":\n"
                            question_field += f"\tQuestion: {question} ?\n"
                            question_field += f"\t\tDescription: {description}\n"
                            question_field += f"\tPossible Values: {possible_values}\n"
                            question_field += f"\tDefault Value: {default_value}\n"

                            section_questions += f"{question_field}\n"
                    else:
                        question = form_element_dict["label"].strip()  # .split(". ")[1]
                        question_field = form_element_dict["ftw_var_name"].strip()
                        question_field_attr = self.model_input[section_name][sub_section_name.split('. ')[1]][
                            question_field]
                        description = question_field_attr['desc']
                        if section_name not in self.model_input:
                            print(f"{section_name} not in Final Design Summary!!!")
                        final_design_summary = question_field_attr['final_design_summary']
                        possible_values = []
                        default_value = ""
                        if "valueFormat" in form_element_dict:
                            format = form_element_dict["valueFormat"]
                        if form_element_dict["type"] in ["radiogroup", "select"]:
                            possible_values = form_element_dict["dataDictFieldUI"]["possibleValues"]
                            default_value = form_element_dict["dataDictFieldUI"]["defaultValue"]
                            if possible_values:
                                possible_values = {val_dict['optionLabel']: val_dict['fieldName'] for val_dict in
                                                   possible_values}
                                possible_values = list(possible_values.keys())
                            else:
                                possible_values = []

                        # question_field += ":\n"
                        # question_field += f"\t\tQuestion: {question} ?\n"
                        # # question_field += f"\t\tDescription: {description}\n"
                        # question_field += f"\t\tPossible Values: {possible_values}\n"
                        # question_field += f"\t\tDefault Value: {default_value}\n"
                        #
                        # section_questions += f"\t{question_field}\n"

                        question_field += ":\n"
                        question_field += f"\tQuestion: {question} ?\n"
                        question_field += f"\tDescription: {description}\n"
                        question_field += f"\tPossible Values: {possible_values}\n"
                        question_field += f"\tDefault Value: {default_value}\n"

                        section_questions += f"{question_field}\n"

        results = self.process_prompts(prompts=prompt_messages)

        final_response = ""
        result_idx_counter = 0
        for page_idx, page_dict in enumerate(survey_studio_question_json["pages"]):
            section_name = page_dict["section_name"]
            if section_name not in self.result_json:
                self.result_json[section_name] = {}
            section_questions = ""
            final_design_summary = ""
            for form_idx, form_element_dict in enumerate(page_dict["form_elements"]):
                format = ""
                if form_element_dict["type"] == "h3" and section_questions:
                    if not mapping_dict.loc[mapping_dict['design_grid_section'] == section_name].empty:
                        user_prompt = mapping_dict.loc[
                            mapping_dict['design_grid_section'] == section_name, "reconstruction_question_prompt"].iloc[
                            0]
                        print("\n")
                        print("---------")
                        print("Question:")
                        print("---------")
                        user_prompt = user_prompt.replace("{sub_section}", section_name).replace(
                            "{final_design_summary}", final_design_summary).replace("{question}", section_questions)
                        print(f"{user_prompt}")
                        print("\n")
                        print("---------")
                        response = results[result_idx_counter]
                        result_idx_counter += 1
                        print("Response:")
                        print("---------")
                        print(response)
                        print("---------")
                        final_response = final_response + "---------\nQuestion:\n---------\n" + user_prompt + "\n---------\nResponse:\n---------\n" + response + "\n---------\n"
                        response = self.format_model_response_for_reconstruct_from_summary(input_str=response)
                        print("Response formatted:")
                        print("---------")
                        print(response)
                        print("---------")
                        self.result_json[section_name][sub_section_name] = response
                        final_response = final_response + "\n---------\nResponse formatted:\n---------\n" + json.dumps(
                            response) + "\n---------\n" + "\n\n\n\n----Separator----\n\n\n\n"
                        print("\n")
                    else:
                        print(f"{section_name} not in mapping dictionary!!!")

                if form_element_dict["type"] == "h3":
                    sub_section_name = form_element_dict["content"]
                    if section_questions:
                        section_questions = ""
                    # TODO: uncomment if sub_section_name name required
                    # section_questions += f"{sub_section_name}:\n"
                if form_element_dict["type"] in ["group", "text", "textarea", "number", "date", "radiogroup", "select"]:
                    if form_element_dict["type"] == "group":
                        for children in form_element_dict["children"]:
                            question = children["label"].strip()  # .split(". ")[1]
                            question_field = children["ftw_var_name"].strip()
                            question_field_attr = self.model_input[section_name][sub_section_name.split('. ')[1]][
                                question_field]
                            description = question_field_attr['desc']
                            if section_name not in self.model_input:
                                print(f"{section_name} not in Final Design Summary!!!")
                            final_design_summary = question_field_attr['final_design_summary']
                            possible_values = []
                            default_value = ""
                            if "valueFormat" in children:
                                format = children["valueFormat"]
                            if children["type"] in ["radiogroup", "select"]:
                                possible_values = children["dataDictFieldUI"]["possibleValues"]
                                default_value = children["dataDictFieldUI"]["defaultValue"]
                                if possible_values:
                                    possible_values = {val_dict['optionLabel']: val_dict['fieldName'] for val_dict in
                                                       possible_values}
                                    possible_values = list(possible_values.keys())
                                else:
                                    possible_values = []

                            # question_field += ":\n"
                            # question_field += f"\t\tQuestion: {question} ?\n"
                            # # question_field += f"\t\tDescription: {description}\n"
                            # question_field += f"\t\tPossible Values: {possible_values}\n"
                            # question_field += f"\t\tDefault Value: {default_value}\n"
                            #
                            # section_questions += f"\t{question_field}\n"

                            question_field += ":\n"
                            question_field += f"\tQuestion: {question} ?\n"
                            question_field += f"\tDescription: {description}\n"
                            question_field += f"\tPossible Values: {possible_values}\n"
                            question_field += f"\tDefault Value: {default_value}\n"

                            section_questions += f"{question_field}\n"
                    else:
                        question = form_element_dict["label"].strip()  # .split(". ")[1]
                        question_field = form_element_dict["ftw_var_name"].strip()
                        question_field_attr = self.model_input[section_name][sub_section_name.split('. ')[1]][
                            question_field]
                        description = question_field_attr['desc']
                        if section_name not in self.model_input:
                            print(f"{section_name} not in Final Design Summary!!!")
                        final_design_summary = question_field_attr['final_design_summary']
                        possible_values = []
                        default_value = ""
                        if "valueFormat" in form_element_dict:
                            format = form_element_dict["valueFormat"]
                        if form_element_dict["type"] in ["radiogroup", "select"]:
                            possible_values = form_element_dict["dataDictFieldUI"]["possibleValues"]
                            default_value = form_element_dict["dataDictFieldUI"]["defaultValue"]
                            if possible_values:
                                possible_values = {val_dict['optionLabel']: val_dict['fieldName'] for val_dict in
                                                   possible_values}
                                possible_values = list(possible_values.keys())
                            else:
                                possible_values = []

                        # question_field += ":\n"
                        # question_field += f"\t\tQuestion: {question} ?\n"
                        # # question_field += f"\t\tDescription: {description}\n"
                        # question_field += f"\t\tPossible Values: {possible_values}\n"
                        # question_field += f"\t\tDefault Value: {default_value}\n"
                        #
                        # section_questions += f"\t{question_field}\n"

                        question_field += ":\n"
                        question_field += f"\tQuestion: {question} ?\n"
                        question_field += f"\tDescription: {description}\n"
                        question_field += f"\tPossible Values: {possible_values}\n"
                        question_field += f"\tDefault Value: {default_value}\n"

                        section_questions += f"{question_field}\n"

        with open(f"{self.result_dir}/final_response.txt", "w") as f:
            f.write(final_response)
        print(f"Time Taken Model run: {time.time() - start_time} seconds")

    def format_model_response_for_detailed_summary(self, input_text):
        if "#### " in input_text:
            print("=============== format_model_response: 1 =================")
            sections = input_text.strip().split("#### ")

            formatted_output = []
            # header = sections[0].strip().replace("### ", "")
            # formatted_output.append(header)

            # section_number = 1

            for section in sections[1:]:
                if section:
                    lines = section.split("\n")
                    title = lines[0].strip()

                    # formatted_output.append(f"{section_number}. {title}")
                    formatted_output.append(f"{title}")

                    for line in lines[1:]:
                        if line.strip():
                            if line.startswith("-"):
                                content = line.replace("- ", "").replace("**", "").strip()
                                formatted_output.append(f"\t* {content}")
                            elif line.startswith("  -"):
                                content = line.replace("- ", "").replace("**", "").strip()
                                formatted_output.append(f"\t\t- {content}")
                            else:
                                content = line.replace("- ", "").replace("**", "").strip()
                                formatted_output.append(f"\t\t\t- {content}")
                    # section_number += 1
        elif "### " in input_text:
            print("=============== format_model_response: 2 =================")
            sections = input_text.strip().split("### ")

            formatted_output = []
            # header = sections[0].strip().replace("### ", "")
            # formatted_output.append(header)
            # section_number = 1

            for section in sections:
                if section:
                    lines = section.split("\n")
                    # title = lines[0].strip()
                    # formatted_output.append(f"\t{section_number}. {title}")

                    for line in lines[1:]:
                        if line.strip():
                            if line.startswith("-"):
                                content = line.replace("- ", "").replace("**", "").strip()
                                formatted_output.append(f"* {content}")
                            else:
                                content = line.replace("- ", "").replace("**", "").strip()
                                formatted_output.append(f"\t- {content}")
                    # section_number += 1
        else:
            print("=============== format_model_response: 3 =================")
            lines = input_text.strip().split("\n")
            formatted_output = []
            for line in lines:
                if line:
                    if line.startswith("-"):
                        content = line.replace("- ", "").replace("**", "").strip()
                        formatted_output.append(f"* {content}")
                    else:
                        content = line.replace("- ", "").replace("**", "").strip()
                        formatted_output.append(f"\t- {content}")
        return "\n".join(formatted_output)

    def format_model_response_for_design_grid_current_provision_summary(self, input_text):
        input_text = input_text.replace("### Summary:\n\n", "").replace("**Summary:**\n\n", "").replace("**Summary:**",
                                                                                                        "").replace(
            "- ", "").strip()
        return input_text

    def format_model_response_for_design_grid_final_summary(self, input_text):
        # input_text = input_text.split("ADJUSTED PLAN DESIGN: ", 1)[1].strip()
        # input_text = input_text.replace("ADJUSTED PLAN DESIGN: ", "").replace("ADJUSTED PLAN DESIGN:", "").strip()

        json_pattern = re.compile(r"{[^}]+}")
        json_match = json_pattern.findall(input_text)

        unique_jsons = set()
        for match in json_match:
            try:
                json_obj = json.loads(match)
                unique_jsons.add(json.dumps(json_obj, sort_keys=True))  # Sort keys to ensure uniqueness
            except json.JSONDecodeError:
                continue
        unique_json_object = [json.loads(obj) for obj in unique_jsons][0]
        return unique_json_object["ADJUSTED PLAN DESIGN"]

    def format_model_response_for_reconstruct_from_summary(self, input_str):
        input_str = input_str.strip('"')
        # json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        json_match = json_pattern.search(input_str)

        try:
            if json_match:
                json_str = json_match.group()
                res_json = json_str
                try:
                    if isinstance(res_json, str):
                        res_json = json.loads(res_json)
                except Exception as e:
                    print(f"Error: {str(e)}")
                return res_json
            else:
                raise ValueError("No JSON object found in the input string")
        except Exception as e:
            print(f"Error: {str(e)}")
            try:
                res_json = (input_str
                            .replace("```json", "")
                            .replace("```", "")
                            .strip())
                json_str = "{" + res_json.strip() + "}"
                print(json_str)
                res_json = json.loads(json_str)
                return res_json
            except Exception as e:
                print(f"Error: {str(e)}")

    def format_output_for_detailed_summary(self):
        start_time = time.time()
        covered_sections = []
        for page_no, fields_dict in self.result_json.items():
            result_json = {"pageNo": int(page_no), "Fields": []}
            for field_name, field_response in fields_dict.items():
                fields_json = {"valueSet": []}
                model_response_json = {}
                if field_name not in covered_sections:
                    fields_json["key"] = field_name
                    fields_json["validation_status"] = None
                    fields_json["is_mandatory"] = True
                    fields_json["confidence_score"] = 100

                    model_response_json["value"] = field_response["value"]
                    model_response_json["coordinates"] = [value / self.points_inches_constant for value in
                                                          field_response["coordinates"]]
                    model_response_json["is_validated"] = False
                    model_response_json["validation_log"] = None
                    model_response_json["source"] = self.model_source

                    fields_json["valueSet"].append(model_response_json)

                    result_json["Fields"].append(fields_json)
                    covered_sections.append(field_name)
            self.final_output["document_summary_result"].append(result_json)
        print(json.dumps(self.final_output))
        print(f"Time Taken Model output format: {time.time() - start_time} seconds")

    def format_output_for_design_grid_current_provision_summary(self):
        start_time = time.time()
        covered_sections = []
        for page_no, fields_dict in self.result_json.items():
            result_json = {"pageNo": int(page_no), "Fields": []}
            for field_name, field_response in fields_dict.items():
                fields_json = {"valueSet": []}
                model_response_json = {}
                if field_name not in covered_sections:
                    fields_json["section"] = field_response["section"]
                    fields_json["section_page_no"] = field_response["section_page_no"]
                    fields_json["sub_section"] = field_response["sub_section"]
                    fields_json["key"] = field_name
                    fields_json["validation_status"] = None
                    fields_json["is_mandatory"] = True
                    fields_json["confidence_score"] = 100

                    model_response_json["value"] = field_response["value"]
                    model_response_json["coordinates"] = []
                    model_response_json["is_validated"] = False
                    model_response_json["validation_log"] = None
                    model_response_json["source"] = self.model_source

                    fields_json["valueSet"].append(model_response_json)

                    result_json["Fields"].append(fields_json)
                    covered_sections.append(field_name)
            self.final_output["design_grid_current_provision_summary_result"].append(result_json)
        print(json.dumps(self.final_output))
        print(f"Time Taken Model output format: {time.time() - start_time} seconds")

    def format_output_for_reconstruct_from_summary(self):
        final_output_json = {}
        with open(f"{self.input_dir}/planDesignSpec.json", "r") as f:
            final_output_json = json.loads(f.read())

        for page_idx, page_dict in enumerate(final_output_json["pages"]):
            section_name = page_dict["section_name"]
            if section_name in self.result_json:
                section_questions_json = self.result_json[section_name]
                for form_idx, form_element_dict in enumerate(page_dict["form_elements"]):
                    if form_element_dict["type"] in ["group", "text", "textarea", "number", "date", "radiogroup",
                                                     "select"]:
                        if form_element_dict["type"] == "group":
                            for child_idx, children in enumerate(form_element_dict["children"]):
                                items = []
                                question = children["label"].strip()  # .split(". ")[1]
                                question_field = children["ftw_var_name"].strip()
                                if "items" in children:
                                    items = {item["label"].strip(): item["value"].strip() for item in children["items"]}
                                if "valueFormat" in children:
                                    format = children["valueFormat"]

                                if section_questions_json and question_field in section_questions_json:
                                    result = section_questions_json[question_field]
                                    if result in ["None", "NONE", "none", "null", "NULL", "Null"]:
                                        result = None
                                    if items:
                                        if result not in items:
                                            result = ""
                                            print(
                                                f"Expected any of {items}, Got {result} for section {section_name} : question {question} !!!")
                                            # if "No" in items:
                                            #     result = "No"
                                            # else:
                                            #     raise ValueError(f"Expected any of {items}, Got {result} for question {question}!!!")
                                        else:
                                            result = items[result]
                                    if result not in [None, ""]:
                                        if "value" in final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                            "children"][child_idx]:
                                            default_value = \
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx]["children"][
                                                child_idx]["value"]
                                            if result and result != default_value:
                                                final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                                    "children"][child_idx]["value"] = result
                                                final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                                    "children"][child_idx]["value_source"] = "System"
                                                final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                                    "children"][child_idx]["confidence_score"] = 100
                                        else:
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx]["children"][
                                                child_idx]["value"] = result
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx]["children"][
                                                child_idx]["value_source"] = "System"
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx]["children"][
                                                child_idx]["confidence_score"] = 100
                                else:
                                    print(f"{question_field} not in result_json under section {section_name}!!!")
                        else:
                            items = []
                            question = form_element_dict["label"].strip()  # .split(". ")[1]
                            question_field = form_element_dict["ftw_var_name"].strip()

                            if "items" in form_element_dict:
                                items = {item["label"].strip(): item["value"].strip() for item in
                                         form_element_dict["items"]}
                            if "valueFormat" in form_element_dict:
                                format = form_element_dict["valueFormat"]

                            if section_questions_json and question_field in section_questions_json:
                                result = section_questions_json[question_field]
                                if result in ["None", "NONE", "none", "null", "NULL", "Null"]:
                                    result = None
                                if items:
                                    if result not in items:
                                        result = ""
                                        print(
                                            f"Expected any of {items}, Got {result} for section {section_name} : question {question} !!!")
                                        # if "No" in items:
                                        #     result = "No"
                                        # else:
                                        #     raise ValueError(f"Expected any of {items}, Got {result} for question {question}!!!")
                                    else:
                                        result = items[result]
                                if result not in [None, ""]:
                                    if "value" in final_output_json["pages"][page_idx]["form_elements"][form_idx]:
                                        default_value = final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                            "value"]
                                        if result and result != default_value:
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                                "value"] = result
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                                "value_source"] = "System"
                                            final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                                "confidence_score"] = 100
                                    else:
                                        final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                            "value"] = result
                                        final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                            "value_source"] = "System"
                                        final_output_json["pages"][page_idx]["form_elements"][form_idx][
                                            "confidence_score"] = 100
                            else:
                                print(f"{question_field} not in result_json under section {section_name}!!!")
            else:
                print(f"{section_name} not in result_json!!!")
        return final_output_json

    def run(self):
        start_time = time.time()

        if self.llm_task == LLMTask.EXTRACT_TEXT:
            self.final_output = {"extract_text_result": []}

            # Step 1: generate model input json
            self.generate_model_input_for_extract_text()
            # Step 2: run model on input json
            self.run_model_for_extract_text()
        elif self.llm_task == LLMTask.DETAILED_SUMMARY:
            self.final_output = {"document_summary_result": []}

            # Step 1: generate model input json
            self.generate_model_input_for_detailed_summary()
            # Step 2: run model on input json
            self.run_model_for_detailed_summary()
            # Step 3: run model on output json
            self.format_output_for_detailed_summary()
        elif self.llm_task == LLMTask.DESIGN_GRID_CURRENT_PROVISION_SUMMARY:
            self.final_output = {"design_grid_current_provision_summary_result": []}

            # Step 1: generate model input json
            self.generate_model_input_for_design_grid_current_provision_summary()
            # Step 2: run model on input json
            self.run_model_for_design_grid_current_provision_summary()
            # Step 3: run model on output json
            self.format_output_for_design_grid_current_provision_summary()
        elif self.llm_task == LLMTask.DESIGN_GRID_FINAL_SUMMARY:
            self.final_output = {"design_grid_final_summary_result": {}}

            # Step 1: run model on input json
            self.run_model_for_design_grid_final_summary()

            self.final_output = {"design_grid_final_summary_result": self.result_json}

            print(json.dumps(self.final_output))

            # with open(f"{self.input_dir}/planDesignGrid.json", "w") as f:
            #     f.write(json.dumps(self.result_json, indent=4, sort_keys=False, default=str))
        elif self.llm_task == LLMTask.RECONSTRUCT_FROM_SUMMARY:
            self.final_output = {"reconstruct_from_summary_result": []}

            # Step 1: generate model input json
            self.generate_model_input_for_reconstruct_from_summary()
            # Step 2: run model on input json
            self.run_model_for_reconstruct_from_summary()
            # Step 3: run model on output json
            final_output_json = self.format_output_for_reconstruct_from_summary()
            self.final_output = {"reconstruct_from_summary_result": final_output_json}

            print(json.dumps(self.final_output))
            # with open(f"{self.input_dir}/planDesignSpec.json", "w") as f:
            #     f.write(json.dumps(self.final_output, indent=4, sort_keys=False, default=str))
        else:
            raise ValueError(f"Unknown LLMTask: {self.llm_task}")

        print(f"Time Taken: {time.time() - start_time} seconds")
        print("\n")

        del self.model
        torch.cuda.empty_cache()
        print("Model unloaded from CUDA memory after function one.")

        with open(f"{self.result_dir}/final_output.json", "w") as f:
            f.write(json.dumps(self.final_output, indent=4, sort_keys=False, default=str))
        print(f"Total Time Taken: {time.time() - start_time} seconds")
        return self.final_output, self.result_dir


if __name__ == "__main__":
    input_dir = "/Users/vyshnavmt/Downloads/RZT/Retirement_AI/exp_RAG/retirement_master_plan/llm_summary_v2/preprocessing_results/TRANSFER_PLAN/FTW/Harrisburg_Tennis_Center_Prior_Adoption_Agreement"
    output_dir = "/Users/vyshnavmt/Downloads/RZT/Retirement_AI/exp_RAG/retirement_master_plan/llm_summary_v2"
    summary_mapping_json_path = "/Users/vyshnavmt/Downloads/RZT/Retirement_AI/exp_RAG/retirement_master_plan/llm_summary_v2/summary_mapping_json"
    form_json_path = "/Users/vyshnavmt/Downloads/RZT/Retirement_AI/exp_RAG/retirement_master_plan/llm_summary_v2"

    llm_obj = LLMModule(input_dir=input_dir, output_dir=output_dir, plan_type=PlanType.TRANSFER_PLAN,
                        doc_type=DocType.FTW, llm_task=LLMTask.EXTRACT_TEXT, source=ModelSource.LLM_MODEL)
    extract_text_output_json, result_dir1 = llm_obj.run()

    llm_obj = LLMModule(input_dir=result_dir1, output_dir=output_dir, plan_type=PlanType.TRANSFER_PLAN,
                        doc_type=DocType.FTW, llm_task=LLMTask.DETAILED_SUMMARY, source=ModelSource.LLM_MODEL)
    summary_output_json, result_dir2 = llm_obj.run()

    llm_obj = LLMModule(input_dir=result_dir2, output_dir=output_dir, plan_type=PlanType.TRANSFER_PLAN,
                        doc_type=DocType.FTW, llm_task=LLMTask.DESIGN_GRID_CURRENT_PROVISION_SUMMARY,
                        source=ModelSource.LLM_MODEL)
    current_provision_summary_json, result_dir3 = llm_obj.run()

    llm_obj = LLMModule(input_dir=form_json_path, output_dir=output_dir, plan_type=PlanType.TRANSFER_PLAN,
                        doc_type=DocType.FTW, llm_task=LLMTask.DESIGN_GRID_FINAL_SUMMARY, source=ModelSource.LLM_MODEL)
    final_design_summary_json, result_dir4 = llm_obj.run()

    llm_obj = LLMModule(input_dir=form_json_path, output_dir=output_dir, plan_type=PlanType.START_UP_PLAN,
                        doc_type=DocType.FTW, llm_task=LLMTask.RECONSTRUCT_FROM_SUMMARY, source=ModelSource.LLM_MODEL)
    reconstruct_from_summary_json, result_dir5 = llm_obj.run()
