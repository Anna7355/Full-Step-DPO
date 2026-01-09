import torch
from utils import tokenizer_setting


class Align_Collator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def collate_fn(self, batch):
        prompts, responses, rewards, rewards_list = [], [], [], []
        b = batch[0]
        responses = [item for item in b["responses"]]
        rewards = b["rewards"]
        rewards_list = b["rewards_list"]
        prompts = [b["prompt"]] * len(b["responses"])

        tokenizer_setting(self.tokenizer, "left")
        prompt_input = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            max_length=self.config.max_prompt_len,
            truncation=True,
        )
        prompt_len = prompt_input["input_ids"].shape[-1]

        tokenizer_setting(self.tokenizer, "right")
        response_input = self.tokenizer(
            responses,
            padding=True,
            return_tensors="pt",
            max_length=self.config.max_response_len,
            truncation=True,
        )
        input_ids = torch.concat(
            [prompt_input["input_ids"], response_input["input_ids"][:, :]], axis=1
        )
        attention_mask = torch.concat(
            [prompt_input["attention_mask"], response_input["attention_mask"][:, :]], axis=1
        )

        weights = torch.zeros_like(response_input["input_ids"]).float()
        if self.config.loss_type == "fullstepdpo":
            for i in range(weights.shape[0]):
                response_input["input_ids"][i, -1] = self.tokenizer.eos_token_id
                indices = torch.where(
                    response_input["input_ids"][i]
                    == self.tokenizer.encode("\nStep")[-1]
                )[0]
                end_index = torch.nonzero(
                    response_input["input_ids"][i] == self.tokenizer.eos_token_id,
                    as_tuple=True,
                )[0][:1]
                indices = (torch.cat([indices, end_index])).tolist()
                last = 0
                for j in range(len(indices)):
                    try:
                        weights[i, last : indices[j]] = rewards_list[i][j]
                        last = indices[j]
                    except:
                        print(rewards_list[i][-1])
                        weights[i, last:] = rewards_list[i][-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, rewards, prompt_len, weights


