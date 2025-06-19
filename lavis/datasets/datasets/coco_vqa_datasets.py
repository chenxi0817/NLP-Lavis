"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import random
from collections import Counter
from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


# class COCOVQADataset(VQADataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         print(ann_paths)
#         print(vis_root)
        
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

#     def __getitem__(self, index):
#         ann = self.annotation[index]

#         image_path = os.path.join(self.vis_root, ann["image"])
#         image = Image.open(image_path).convert("RGB")

#         image = self.vis_processor(image)
#         question = self.text_processor(ann["question"])

#         answer_weight = {}
#         for answer in ann["answer"]:
#             if answer in answer_weight.keys():
#                 answer_weight[answer] += 1 / len(ann["answer"])
#             else:
#                 answer_weight[answer] = 1 / len(ann["answer"])

#         answers = list(answer_weight.keys())
#         weights = list(answer_weight.values())

#         return {
#             "image": image,
#             "text_input": question,
#             "answers": answers,
#             "weights": weights,
#         }

class COCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        这个初始化方法现在只调用父类来加载数据。
        LAVIS提供的vqa_train.json等文件已经是合并好的，无需我们再进行复杂的匹配。
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # =================================================================

        # 设置是否开启调试模式
        is_debug_mode = False  # 在这里切换 True/False
        debug_sample_count = 20480  # 用于调试的样本数量

        is_train_split = any("vqa_train.json" in path for path in ann_paths)

        if is_train_split and is_debug_mode:
            # 确保annotation列表不为空且长度大于调试数量
            if self.annotation and len(self.annotation) > debug_sample_count:
                original_size = len(self.annotation)
                
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"!!! DEBUG MODE ON: Slicing training dataset to first {debug_sample_count} samples (out of {original_size}).")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                # 直接对 self.annotation 列表进行切片
                self.annotation = self.annotation[:debug_sample_count]
        
        # =================================================================

    def __getitem__(self, index):
        """
        重写__getitem__方法，以处理包含10个答案的列表，并返回一个单一的、最可靠的答案。
        """
        # ann 是一个字典，格式为: {"question_id": ..., "question": ..., "answer": ["ans1", "ans2", ...]}
        ann = self.annotation[index]

        # 构造完整的图片路径
        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # 如果找不到图片，打印警告并跳过这个样本
            logging.warning(f"Image not found at {image_path}. Skipping sample.")
            return None

        # --- 从10个答案中找出“众数”作为唯一答案 ---
        # ann["answer"] 是一个包含10个字符串的列表
        ground_truths = ann["answer"]
        # 使用Counter计算每个答案出现的次数
        answer_counts = Counter(ground_truths)
        # most_common(1) 返回一个列表，如 [('yes', 8)]，我们取第一个元组的第一个元素
        most_frequent_answer = answer_counts.most_common(1)[0][0]

        # --- 数据处理 ---
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        # --- 返回适合生成式微调的、干净的数据格式 ---
        return {
            "image": image,
            "text_input": question,
            "answer": most_frequent_answer,  # 使用我们精心挑选的单一、最可靠的答案
            "question_id": ann["question_id"]
        }

    def collater(self, samples):
        """
        自定义collater，用于处理我们简化的__getitem__输出。
        这个方法将覆盖父类VQADataset中不兼容的旧方法。
        """
        # 过滤掉在__getitem__中可能因为找不到图片而返回的None样本
        samples = [s for s in samples if s is not None]
        if not samples:
            # 如果一个批次的所有样本都无效，返回None
            return None

        image_list, question_list, answer_list, question_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["answer"])
            question_id_list.append(sample["question_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list, # 注意：这里的键是 'answer'，与我们 vqa.py 中 train_step 的期望完全匹配
            "question_id": torch.tensor(question_id_list, dtype=torch.int)
        }


class COCOVQAInstructDataset(COCOVQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = random.choice(data["answers"])
        return data

    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data

    

class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
